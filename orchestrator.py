"""
orchestrator.py - realtime ASR -> MT -> TTS pipeline

Current design:
- ASR: Qwen realtime ASR or local sherpa-onnx zipformer
- MT: DeepSeek API or local translation API
- TTS: speak() imported directly from main.py

TTS playback is serialized through a worker queue so one utterance does not
interrupt another while ASR continues listening.
"""

import base64
import os
import queue
import sys
import threading

import numpy as np
import requests
import sherpa_onnx
import sounddevice as sd
import torch
from dotenv import load_dotenv
from pyrnnoise import RNNoise

from main import speak as tts_speak

load_dotenv()


USE_LOCAL_MT = os.getenv("USE_LOCAL_MT", "false").lower() == "true"
USE_QWEN_ASR_API = os.getenv("USE_QWEN_ASR_API", "false").lower() == "true"

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
if not USE_LOCAL_MT and not DEEPSEEK_API_KEY:
    raise ValueError("DEEPSEEK_API_KEY is not set in .env")

DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")

DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")

QWEN_ASR_MODEL = os.getenv(
    "QWEN_ASR_MODEL",
    "qwen3-asr-flash-realtime-2025-10-27",
)
QWEN_ASR_URL = os.getenv(
    "QWEN_ASR_URL",
    "wss://dashscope.aliyuncs.com/api-ws/v1/realtime",
)
QWEN_ASR_LANGUAGE = os.getenv("QWEN_ASR_LANGUAGE", "zh")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "zipformer")

SAMPLE_RATE = 16000
FRAME_SIZE = 512
CHANNELS = 1

MT_URL = os.getenv("MT_URL", "http://127.0.0.1:8000/translate")
MT_SOURCE_LANG = os.getenv("MT_SOURCE_LANG", "zh")
MT_TARGET_LANG = os.getenv("MT_TARGET_LANG", "en")
MT_TIMEOUT_SEC = int(os.getenv("MT_TIMEOUT_SEC", "8"))

VAD_THRESHOLD = 0.40
MAX_SILENCE_FRAMES = 30

LANGUAGE_NAME_MAP = {
    "zh": "Chinese",
    "en": "English",
    "ja": "Japanese",
    "ko": "Korean",
    "fr": "French",
    "de": "German",
    "es": "Spanish",
    "ru": "Russian",
    "pt": "Portuguese",
    "it": "Italian",
    "ar": "Arabic",
    "hi": "Hindi",
    "id": "Indonesian",
    "th": "Thai",
    "vi": "Vietnamese",
}


class LocalStreamingASR:
    def __init__(self):
        tokens_path = os.path.join(MODEL_PATH, "tokens.txt")
        encoder_path = os.path.join(MODEL_PATH, "encoder-epoch-99-avg-1.int8.onnx")
        decoder_path = os.path.join(MODEL_PATH, "decoder-epoch-99-avg-1.onnx")
        joiner_path = os.path.join(MODEL_PATH, "joiner-epoch-99-avg-1.int8.onnx")

        print(
            "[ASR model] "
            f"provider=sherpa-onnx | model_path={MODEL_PATH} | "
            "decoding_method=greedy_search"
        )
        print(
            "[VAD model] "
            "repo=snakers4/silero-vad | model=silero_vad | "
            f"threshold={VAD_THRESHOLD} | max_silence_frames={MAX_SILENCE_FRAMES}"
        )

        self.recognizer = sherpa_onnx.OnlineRecognizer.from_transducer(
            tokens=tokens_path,
            encoder=encoder_path,
            decoder=decoder_path,
            joiner=joiner_path,
            num_threads=4,
            sample_rate=SAMPLE_RATE,
            feature_dim=80,
            decoding_method="greedy_search",
        )
        self.stream = self.recognizer.create_stream()
        self.rnnoise = RNNoise(sample_rate=SAMPLE_RATE)
        self.vad_model, _ = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            force_reload=False,
        )
        self.vad_model.eval()

        self.last_text = ""
        self.silence_frames = 0

    def _denoise(self, audio: np.ndarray) -> np.ndarray:
        try:
            chunks = []
            for i in range(0, 480, 160):
                res = self.rnnoise.process(audio[i : i + 160])
                chunks.append(res[0] if isinstance(res, tuple) else res)
            return np.concatenate(chunks + [audio[480:]]).astype(np.float32)
        except Exception:
            return audio

    def _vad_prob(self, audio: np.ndarray) -> float:
        tensor = torch.from_numpy(audio).unsqueeze(0)
        with torch.no_grad():
            return self.vad_model(tensor, SAMPLE_RATE).item()

    def _reset(self):
        self.stream = self.recognizer.create_stream()
        self.last_text = ""
        self.silence_frames = 0

    def process_audio(self, audio: np.ndarray) -> str:
        denoised = self._denoise(audio)
        voice_prob = self._vad_prob(denoised)

        if voice_prob > VAD_THRESHOLD:
            self.stream.accept_waveform(SAMPLE_RATE, denoised)
            self.silence_frames = 0
        else:
            self.silence_frames += 1

        while self.recognizer.is_ready(self.stream):
            self.recognizer.decode_stream(self.stream)

        current = self.recognizer.get_result(self.stream).strip()
        if current and current != self.last_text:
            self.last_text = current

        if self.last_text and self.silence_frames >= MAX_SILENCE_FRAMES:
            final = self.last_text
            print(f"\n[ASR final  ]: {final}")
            self._reset()
            return final

        return ""

    def close(self) -> None:
        return None


class _QwenASRCallback:
    def __init__(self, final_queue: queue.Queue[str]):
        self.final_queue = final_queue

    def on_open(self):
        return None

    def on_close(self, code, msg):
        return None

    def on_event(self, response):
        try:
            event_type = response.get("type", "")
            if event_type == "conversation.item.input_audio_transcription.completed":
                transcript = response.get("transcript", "").strip()
                if transcript:
                    self.final_queue.put(transcript)
        except Exception as exc:
            print(f"[ASR] Qwen callback error: {exc}")


class QwenStreamingASR:
    def __init__(self):
        if not DASHSCOPE_API_KEY:
            raise ValueError("DASHSCOPE_API_KEY is required when USE_QWEN_ASR_API=true")
        try:
            import dashscope
            from dashscope.audio.qwen_omni import MultiModality, OmniRealtimeConversation
            from dashscope.audio.qwen_omni.omni_realtime import TranscriptionParams
        except ImportError as exc:
            raise ImportError("dashscope>=1.25.6 is required for Qwen ASR") from exc

        dashscope.api_key = DASHSCOPE_API_KEY
        self._final_queue: queue.Queue[str] = queue.Queue()
        self._callback = _QwenASRCallback(self._final_queue)
        self._conversation = OmniRealtimeConversation(
            model=QWEN_ASR_MODEL,
            url=QWEN_ASR_URL,
            callback=self._callback,
        )
        self._conversation.connect()

        params = TranscriptionParams(
            language=QWEN_ASR_LANGUAGE,
            sample_rate=SAMPLE_RATE,
            input_audio_format="pcm",
        )
        self._conversation.update_session(
            output_modalities=[MultiModality.TEXT],
            enable_input_audio_transcription=True,
            transcription_params=params,
        )

        print(
            "[ASR model] "
            f"provider=qwen_api | model={QWEN_ASR_MODEL} | language={QWEN_ASR_LANGUAGE}"
        )

    def process_audio(self, audio: np.ndarray) -> str:
        pcm = np.clip(audio, -1.0, 1.0)
        pcm16 = (pcm * 32767).astype(np.int16)
        audio_b64 = base64.b64encode(pcm16.tobytes()).decode("ascii")
        self._conversation.append_audio(audio_b64)
        try:
            final = self._final_queue.get_nowait()
            print(f"\n[ASR final  ]: {final}")
            return final
        except queue.Empty:
            return ""

    def close(self) -> None:
        try:
            self._conversation.end_session()
        except Exception:
            pass
        try:
            self._conversation.close()
        except Exception:
            pass


def build_asr_backend():
    if USE_QWEN_ASR_API:
        try:
            asr = QwenStreamingASR()
            print("[ASR route] primary=qwen_api | fallback=zipformer")
            return asr
        except Exception as exc:
            print(f"[ASR] Qwen ASR unavailable, fallback to local zipformer: {exc}")
    asr = LocalStreamingASR()
    print("[ASR route] primary=zipformer")
    return asr


def _translate_local(text: str) -> str | None:
    try:
        print(
            "[MT request] "
            f"url={MT_URL} | source={MT_SOURCE_LANG} | target={MT_TARGET_LANG} | "
            f"timeout={MT_TIMEOUT_SEC}s"
        )
        resp = requests.post(
            MT_URL,
            json={
                "text": text,
                "source_lang": MT_SOURCE_LANG,
                "target_lang": MT_TARGET_LANG,
            },
            timeout=MT_TIMEOUT_SEC,
        )
        resp.raise_for_status()
        result = resp.json().get("translated_text", "").strip()
        if result:
            print(f"[MT  ]: {result}")
        return result or None
    except requests.exceptions.Timeout:
        print("[MT] Timed out - skipping utterance.")
        return None
    except Exception as exc:
        print(f"[MT] Error: {exc}")
        return None


def _translate_deepseek(text: str) -> str | None:
    try:
        source_name = LANGUAGE_NAME_MAP.get(MT_SOURCE_LANG, MT_SOURCE_LANG)
        target_name = LANGUAGE_NAME_MAP.get(MT_TARGET_LANG, MT_TARGET_LANG)
        headers = {
            "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": DEEPSEEK_MODEL,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a professional interpreter. "
                        f"Translate from {source_name} to {target_name}. "
                        "Output only the translation."
                    ),
                },
                {
                    "role": "user",
                    "content": text,
                },
            ],
            "temperature": 0.3,
        }

        resp = requests.post(
            f"{DEEPSEEK_BASE_URL}/chat/completions",
            headers=headers,
            json=payload,
            timeout=MT_TIMEOUT_SEC,
        )
        resp.raise_for_status()

        data = resp.json()
        result = data["choices"][0]["message"]["content"].strip()
        if result:
            print(f"[MT  ]: {result}")
        return result or None
    except requests.exceptions.Timeout:
        print("[MT] Timed out - skipping utterance.")
        return None
    except Exception as exc:
        print(f"[MT] Exception: {exc}")
        return None


def translate(text: str) -> str | None:
    if USE_LOCAL_MT:
        return _translate_local(text)
    return _translate_deepseek(text)


_audio_q: queue.Queue[np.ndarray] = queue.Queue()
_tts_q: queue.Queue[tuple[str, str]] = queue.Queue()


def _audio_callback(indata, frames, time_info, status):
    if status:
        print(f"[Audio] {status}", file=sys.stderr)
    _audio_q.put(indata[:, 0].copy())


def _run_tts(text: str, lang: str) -> None:
    try:
        tts_speak(text, lang=lang)
    except Exception as exc:
        print(f"[TTS thread] crashed: {exc}")


def _tts_worker() -> None:
    while True:
        text, lang = _tts_q.get()
        try:
            _run_tts(text, lang)
        finally:
            _tts_q.task_done()


def main():
    print("ASR -> MT -> TTS pipeline starting...\n")
    print(
        "[Pipeline config] "
        f"audio_sample_rate={SAMPLE_RATE} | frame_size={FRAME_SIZE} | channels={CHANNELS} | "
        f"mt_target_lang={MT_TARGET_LANG}"
    )

    if USE_LOCAL_MT:
        print(
            "[MT model ] "
            f"provider=local_api | url={MT_URL} | source={MT_SOURCE_LANG} | target={MT_TARGET_LANG}"
        )
    else:
        print(f"[MT model ] provider=deepseek | model={DEEPSEEK_MODEL}")

    asr = build_asr_backend()
    threading.Thread(target=_tts_worker, daemon=True, name="tts-worker").start()
    print("\nSystem ready - start speaking!\n")

    try:
        with sd.InputStream(
            samplerate=SAMPLE_RATE,
            blocksize=FRAME_SIZE,
            channels=CHANNELS,
            dtype="float32",
            callback=_audio_callback,
        ):
            while True:
                chunk = _audio_q.get()
                final = asr.process_audio(chunk)
                if not final:
                    continue

                translation = translate(final)
                if not translation:
                    continue

                _tts_q.put((translation, MT_TARGET_LANG))
    except KeyboardInterrupt:
        print("\nPipeline stopped.")
    finally:
        asr.close()


if __name__ == "__main__":
    main()
