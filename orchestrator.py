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
import time
from collections import deque
from dataclasses import dataclass

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
AUDIO_INPUT_LATENCY = os.getenv("ASR_INPUT_LATENCY", "high")
AUDIO_QUEUE_MAX_CHUNKS = int(os.getenv("ASR_AUDIO_QUEUE_MAX_CHUNKS", "64"))
ASR_RESULT_QUEUE_MAX = int(os.getenv("ASR_RESULT_QUEUE_MAX", "16"))
TTS_QUEUE_MAX = int(os.getenv("TTS_QUEUE_MAX", "16"))

MT_URL = os.getenv("MT_URL", "http://127.0.0.1:8000/translate")
MT_SOURCE_LANG = os.getenv("MT_SOURCE_LANG", "zh")
MT_TARGET_LANG = os.getenv("MT_TARGET_LANG", "en")
MT_TIMEOUT_SEC = int(os.getenv("MT_TIMEOUT_SEC", "8"))

RNNOISE_FRAME_SIZE = 160
HPF_ALPHA = 0.97
VAD_START_THRESHOLD = 0.55
VAD_END_THRESHOLD = 0.35
MIN_SPEECH_START_FRAMES = 2
MIN_SPEECH_FRAMES = 6
ENERGY_THRESHOLD = 0.008
ENERGY_RELEASE_RATIO = 0.65
PRE_SPEECH_FRAMES = 8
VAD_SMOOTHING_FRAMES = 5
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


@dataclass
class _FrontEndDecision:
    feed_chunks: list[np.ndarray]
    should_finalize: bool = False
    should_reset: bool = False
    speech_frames: int = 0
    vad_prob: float = 0.0
    rms: float = 0.0


class _SpeechFrontEnd:
    def __init__(self):
        self.rnnoise = RNNoise(sample_rate=SAMPLE_RATE)
        self.vad_model, _ = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            force_reload=False,
        )
        self.vad_model.eval()
        self._pre_roll: deque[np.ndarray] = deque(maxlen=PRE_SPEECH_FRAMES)
        self._vad_history: deque[float] = deque(maxlen=VAD_SMOOTHING_FRAMES)
        self._in_speech = False
        self._speech_frames = 0
        self._start_frames = 0
        self._silence_frames = 0
        self._hpf_prev_x = 0.0
        self._hpf_prev_y = 0.0

    def _high_pass(self, audio: np.ndarray) -> np.ndarray:
        if audio.size == 0:
            return audio

        samples = audio.astype(np.float32, copy=False)
        diff = np.empty_like(samples, dtype=np.float64)
        diff[0] = float(samples[0]) - self._hpf_prev_x + (HPF_ALPHA * self._hpf_prev_y)
        if samples.size > 1:
            diff[1:] = np.diff(samples.astype(np.float64, copy=False))

        powers = np.power(HPF_ALPHA, np.arange(samples.size, dtype=np.float64))
        filtered = powers * np.cumsum(diff / powers)

        self._hpf_prev_x = float(samples[-1])
        self._hpf_prev_y = float(filtered[-1])
        return np.clip(filtered, -1.0, 1.0).astype(np.float32, copy=False)

    def _denoise(self, audio: np.ndarray) -> np.ndarray:
        if audio.size == 0:
            return audio

        chunks: list[np.ndarray] = []
        for start in range(0, len(audio), RNNOISE_FRAME_SIZE):
            block = audio[start : start + RNNOISE_FRAME_SIZE].astype(np.float32, copy=False)
            original_len = len(block)
            if original_len == 0:
                continue
            if original_len < RNNOISE_FRAME_SIZE:
                block = np.pad(block, (0, RNNOISE_FRAME_SIZE - original_len))
            try:
                processed = self.rnnoise.process(block)
                processed_block = processed[0] if isinstance(processed, tuple) else processed
                chunks.append(np.asarray(processed_block[:original_len], dtype=np.float32))
            except Exception:
                chunks.append(block[:original_len].astype(np.float32, copy=False))
        if not chunks:
            return np.empty(0, dtype=np.float32)
        return np.concatenate(chunks, dtype=np.float32)

    def _vad_prob(self, audio: np.ndarray) -> float:
        tensor = torch.from_numpy(audio.astype(np.float32, copy=False)).unsqueeze(0)
        with torch.no_grad():
            return float(self.vad_model(tensor, SAMPLE_RATE).item())

    def _reset_utterance_state(self) -> None:
        self._pre_roll.clear()
        self._vad_history.clear()
        self._in_speech = False
        self._speech_frames = 0
        self._start_frames = 0
        self._silence_frames = 0

    def process_audio(self, audio: np.ndarray) -> _FrontEndDecision:
        filtered = self._high_pass(audio)
        denoised = self._denoise(filtered)
        if denoised.size == 0:
            return _FrontEndDecision(feed_chunks=[])

        rms = float(np.sqrt(np.mean(np.square(denoised), dtype=np.float32)))
        vad_prob = self._vad_prob(denoised)
        self._vad_history.append(vad_prob)
        smoothed_prob = float(sum(self._vad_history) / len(self._vad_history))

        start_gate = smoothed_prob >= VAD_START_THRESHOLD and rms >= ENERGY_THRESHOLD
        continue_gate = (
            smoothed_prob >= VAD_END_THRESHOLD
            and rms >= (ENERGY_THRESHOLD * ENERGY_RELEASE_RATIO)
        )

        if not self._in_speech:
            self._pre_roll.append(denoised.copy())
            self._start_frames = self._start_frames + 1 if start_gate else 0
            if self._start_frames < MIN_SPEECH_START_FRAMES:
                return _FrontEndDecision(feed_chunks=[], vad_prob=smoothed_prob, rms=rms)

            self._in_speech = True
            self._speech_frames = len(self._pre_roll)
            self._silence_frames = 0
            self._start_frames = 0
            feed_chunks = list(self._pre_roll)
            self._pre_roll.clear()
            return _FrontEndDecision(
                feed_chunks=feed_chunks,
                speech_frames=self._speech_frames,
                vad_prob=smoothed_prob,
                rms=rms,
            )

        self._speech_frames += 1
        if continue_gate:
            self._silence_frames = 0
        else:
            self._silence_frames += 1

        should_finalize = self._silence_frames >= MAX_SILENCE_FRAMES
        speech_frames = self._speech_frames
        if should_finalize:
            self._reset_utterance_state()

        return _FrontEndDecision(
            feed_chunks=[denoised],
            should_finalize=should_finalize,
            should_reset=should_finalize and speech_frames < MIN_SPEECH_FRAMES,
            speech_frames=speech_frames,
            vad_prob=smoothed_prob,
            rms=rms,
        )


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
            f"start_threshold={VAD_START_THRESHOLD} | "
            f"end_threshold={VAD_END_THRESHOLD} | "
            f"energy_threshold={ENERGY_THRESHOLD:.4f} | "
            f"max_silence_frames={MAX_SILENCE_FRAMES}"
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
        self.frontend = _SpeechFrontEnd()

        self.last_text = ""

    def _reset(self):
        self.stream = self.recognizer.create_stream()
        self.last_text = ""

    def process_audio(self, audio: np.ndarray) -> str:
        decision = self.frontend.process_audio(audio)

        for chunk in decision.feed_chunks:
            self.stream.accept_waveform(SAMPLE_RATE, chunk)
            while self.recognizer.is_ready(self.stream):
                self.recognizer.decode_stream(self.stream)

        if decision.feed_chunks:
            current = self.recognizer.get_result(self.stream).strip()
            if current and current != self.last_text:
                self.last_text = current

        if decision.should_finalize:
            final = self.recognizer.get_result(self.stream).strip() or self.last_text
            self._reset()
            if decision.should_reset:
                return ""
            if final:
                print(f"\n[ASR final  ]: {final}")
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
        self._dashscope = dashscope
        self._MultiModality = MultiModality
        self._OmniRealtimeConversation = OmniRealtimeConversation
        self._TranscriptionParams = TranscriptionParams
        self._final_queue: queue.Queue[str] = queue.Queue()
        self._callback = _QwenASRCallback(self._final_queue)
        self.frontend = _SpeechFrontEnd()
        self._conversation = None
        self._connect_conversation()

        print(
            "[ASR model] "
            f"provider=qwen_api | model={QWEN_ASR_MODEL} | language={QWEN_ASR_LANGUAGE}"
        )

    def _connect_conversation(self) -> None:
        self._conversation = self._OmniRealtimeConversation(
            model=QWEN_ASR_MODEL,
            url=QWEN_ASR_URL,
            callback=self._callback,
        )
        self._conversation.connect()

        params = self._TranscriptionParams(
            language=QWEN_ASR_LANGUAGE,
            sample_rate=SAMPLE_RATE,
            input_audio_format="pcm",
        )
        self._conversation.update_session(
            output_modalities=[self._MultiModality.TEXT],
            enable_input_audio_transcription=True,
            transcription_params=params,
        )

    def _reconnect_conversation(self) -> None:
        print("[ASR] Qwen websocket closed, reconnecting...")
        try:
            if self._conversation is not None:
                self._conversation.close()
        except Exception:
            pass
        self._connect_conversation()

    def process_audio(self, audio: np.ndarray) -> str:
        decision = self.frontend.process_audio(audio)
        for chunk in decision.feed_chunks:
            pcm = np.clip(chunk, -1.0, 1.0)
            pcm16 = (pcm * 32767).astype(np.int16)
            audio_b64 = base64.b64encode(pcm16.tobytes()).decode("ascii")
            try:
                self._conversation.append_audio(audio_b64)
            except Exception as exc:
                message = str(exc).lower()
                if "closed" not in message and "lost" not in message and "timeout" not in message:
                    raise
                self._reconnect_conversation()
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
                        "You are a strict translation engine, not a chatbot. "
                        f"Translate the user's text from {source_name} to {target_name}. "
                        "Return only the translated text. "
                        "Do not answer the user. "
                        "Do not continue the conversation. "
                        "Do not explain anything. "
                        "Do not add quotation marks or notes. "
                        "If the input is already in the target language, return it unchanged. "
                        "If the input is a fragment, translate only that fragment."
                    ),
                },
                {
                    "role": "user",
                    "content": text,
                },
            ],
            "temperature": 0.0,
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


_audio_q: queue.Queue[np.ndarray] = queue.Queue(maxsize=AUDIO_QUEUE_MAX_CHUNKS)
_asr_text_q: queue.Queue[str] = queue.Queue(maxsize=ASR_RESULT_QUEUE_MAX)
_tts_q: queue.Queue[tuple[str, str]] = queue.Queue(maxsize=TTS_QUEUE_MAX)
_audio_overflow_count = 0
_audio_drop_count = 0
_queue_drop_counts = {
    "asr_text": 0,
    "tts": 0,
}


def _put_latest(queue_obj: queue.Queue, item, label: str) -> None:
    try:
        queue_obj.put_nowait(item)
        return
    except queue.Full:
        pass

    _queue_drop_counts[label] = _queue_drop_counts.get(label, 0) + 1

    try:
        queue_obj.get_nowait()
    except queue.Empty:
        pass

    try:
        queue_obj.put_nowait(item)
    except queue.Full:
        pass

    drop_count = _queue_drop_counts[label]
    if drop_count == 1 or drop_count % 10 == 0:
        print(
            f"[Queue] {label} queue full, dropping oldest item (drop_count={drop_count})",
            file=sys.stderr,
        )


def _audio_callback(indata, frames, time_info, status):
    global _audio_overflow_count, _audio_drop_count
    if status:
        _audio_overflow_count += 1
        print(f"[Audio] {status} | overflow_count={_audio_overflow_count}", file=sys.stderr)

    chunk = indata[:, 0].copy()
    try:
        _audio_q.put_nowait(chunk)
    except queue.Full:
        _audio_drop_count += 1
        try:
            _audio_q.get_nowait()
        except queue.Empty:
            pass
        try:
            _audio_q.put_nowait(chunk)
        except queue.Full:
            pass
        if _audio_drop_count == 1 or _audio_drop_count % 10 == 0:
            print(
                "[Audio] capture queue full, dropping oldest buffered chunk "
                f"(drop_count={_audio_drop_count})",
                file=sys.stderr,
            )


def _run_tts(text: str, lang: str) -> None:
    try:
        tts_speak(text, lang=lang)
    except Exception as exc:
        print(f"[TTS thread] crashed: {exc}")


def _asr_worker(asr, stop_event: threading.Event) -> None:
    while not stop_event.is_set():
        try:
            chunk = _audio_q.get(timeout=0.2)
        except queue.Empty:
            continue

        try:
            try:
                final = asr.process_audio(chunk)
                if final:
                    _put_latest(_asr_text_q, final, "asr_text")
            except Exception as exc:
                print(f"[ASR worker] recoverable error: {exc}", file=sys.stderr)
                time.sleep(0.2)
        finally:
            _audio_q.task_done()


def _mt_worker(stop_event: threading.Event) -> None:
    while not stop_event.is_set():
        try:
            text = _asr_text_q.get(timeout=0.2)
        except queue.Empty:
            continue

        try:
            translation = translate(text)
            if translation:
                _put_latest(_tts_q, (translation, MT_TARGET_LANG), "tts")
        finally:
            _asr_text_q.task_done()


def _tts_worker(stop_event: threading.Event) -> None:
    while not stop_event.is_set():
        try:
            text, lang = _tts_q.get(timeout=0.2)
        except queue.Empty:
            continue

        try:
            _run_tts(text, lang)
        finally:
            _tts_q.task_done()


def main():
    print("ASR -> MT -> TTS pipeline starting...\n")
    print(
        "[Pipeline config] "
        f"audio_sample_rate={SAMPLE_RATE} | frame_size={FRAME_SIZE} | channels={CHANNELS} | "
        f"input_latency={AUDIO_INPUT_LATENCY} | audio_queue_max_chunks={AUDIO_QUEUE_MAX_CHUNKS} | "
        f"asr_result_queue_max={ASR_RESULT_QUEUE_MAX} | tts_queue_max={TTS_QUEUE_MAX} | "
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
    stop_event = threading.Event()
    workers = [
        threading.Thread(
            target=_asr_worker,
            args=(asr, stop_event),
            daemon=True,
            name="asr-worker",
        ),
        threading.Thread(
            target=_mt_worker,
            args=(stop_event,),
            daemon=True,
            name="mt-worker",
        ),
        threading.Thread(
            target=_tts_worker,
            args=(stop_event,),
            daemon=True,
            name="tts-worker",
        ),
    ]

    for worker in workers:
        worker.start()

    print("\nSystem ready - start speaking!\n")

    try:
        with sd.InputStream(
            samplerate=SAMPLE_RATE,
            blocksize=FRAME_SIZE,
            channels=CHANNELS,
            dtype="float32",
            latency=AUDIO_INPUT_LATENCY,
            callback=_audio_callback,
        ):
            while True:
                time.sleep(0.5)
    except KeyboardInterrupt:
        print("\nPipeline stopped.")
    finally:
        stop_event.set()
        for worker in workers:
            worker.join(timeout=1.0)
        asr.close()


if __name__ == "__main__":
    main()
