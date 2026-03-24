"""
orchestrator.py - ASR -> MT -> TTS pipeline

TTS integration fix
-------------------
Previously: orchestrator used TTSClient which published to Redis and waited
for audio bytes back, but main.py never sent anything back, causing timeout.

Fix: import speak() from main.py directly. No Redis, no network, no timeout.
TTS requests are queued and played serially so one utterance does not interrupt
another while ASR keeps listening.
"""

import os
import queue
import sys
import threading

import numpy as np
import requests
import sherpa_onnx
import sounddevice as sd
import torch
from pyrnnoise import RNNoise

from main import speak as tts_speak

###Deepseek
from dotenv import load_dotenv
load_dotenv()

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
if not DEEPSEEK_API_KEY:
    raise ValueError("DEEPSEEK_API_KEY is not set in .env")

DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")

###Qwen
"""
from dotenv import load_dotenv
import dashscope
from dashscope import Generation

# 加载 .env 文件
load_dotenv()

api_key = os.getenv("DASHSCOPE_API_KEY")
if not api_key:
    raise ValueError("DASHSCOPE_API_KEY is not set in .env")

dashscope.api_key = api_key
"""



BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "zipformer")

SAMPLE_RATE = 16000
FRAME_SIZE = 512
CHANNELS = 1

MT_URL = "http://127.0.0.1:8000/translate"
MT_SOURCE_LANG = "zh"
MT_TARGET_LANG = "en"
MT_TIMEOUT_SEC = 8

VAD_THRESHOLD = 0.40
MAX_SILENCE_FRAMES = 30


class StreamingASR:
    def __init__(self):
        tokens_path = os.path.join(MODEL_PATH, "tokens.txt")
        encoder_path = os.path.join(MODEL_PATH, "encoder-epoch-99-avg-1.int8.onnx")
        decoder_path = os.path.join(MODEL_PATH, "decoder-epoch-99-avg-1.onnx")
        joiner_path = os.path.join(MODEL_PATH, "joiner-epoch-99-avg-1.int8.onnx")

        print(f"Loading ASR model from: {MODEL_PATH}")
        print(
            "[ASR choice] "
            f"tokens={tokens_path} | encoder={encoder_path} | "
            f"decoder={decoder_path} | joiner={joiner_path}"
        )
        print(
            "[ASR config] "
            f"sample_rate={SAMPLE_RATE} | feature_dim=80 | num_threads=4 | "
            "decoding_method=greedy_search"
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

        print("Initialising RNNoise ...")
        self.rnnoise = RNNoise(sample_rate=SAMPLE_RATE)
        print(f"[RNNoise choice] sample_rate={SAMPLE_RATE}")

        print("Loading Silero VAD ...")
        print(
            "[VAD choice] "
            "repo=snakers4/silero-vad | model=silero_vad | "
            f"threshold={VAD_THRESHOLD} | max_silence_frames={MAX_SILENCE_FRAMES}"
        )
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
            print(f"\r[ASR partial]: {current}   ", end="", flush=True)
            self.last_text = current

        if self.last_text and self.silence_frames >= MAX_SILENCE_FRAMES:
            final = self.last_text
            print(f"\n[ASR final  ]: {final}")
            self._reset()
            return final

        return ""

###这部分是用本地模型
"""""
def translate(text: str) -> str | None:
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
"""""
###这是Qwen的api
""""
def translate(text: str) -> str | None:
    try:
        response = Generation.call(
            model="qwen-turbo",
            messages=[
                {"role": "system", "content": "You are a professional interpreter. Translate Chinese to English. Output only the translation."},
                {"role": "user", "content": text}
            ],
            result_format="message",
        )

        if response.status_code == 200:
            result = response.output.choices[0].message.content.strip()
            print(f"[MT  ]: {result}")
            return result

        print(f"[MT] API error: {response.status_code}")
        return None

    except Exception as e:
        print(f"[MT] Exception: {e}")
        return None
"""

###这是Deepseek的api
def translate(text: str) -> str | None:
    try:
        headers = {
            "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": DEEPSEEK_MODEL,
            "messages": [
                {
                    "role": "system",
                    "content": "You are a professional interpreter. Translate Chinese to English. Output only the translation."
                },
                {
                    "role": "user",
                    "content": text
                }
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


_audio_q: queue.Queue[np.ndarray] = queue.Queue()
_tts_q: queue.Queue[tuple[str, str]] = queue.Queue()


def _audio_callback(indata, frames, time_info, status):
    if status:
        print(status, file=sys.stderr)
    _audio_q.put(indata[:, 0].copy())


def _run_tts(text: str, lang: str) -> None:
    print(f"[TTS thread] starting | lang={lang} | text={text[:60]}")
    try:
        tts_speak(text, lang=lang)
        print("[TTS thread] finished")
    except Exception as exc:
        print(f"[TTS thread] crashed: {exc}")


def _tts_worker() -> None:
    print("[TTS worker] ready | mode=serial_queue")
    while True:
        text, lang = _tts_q.get()
        print(
            f"[TTS worker] dequeued | lang={lang} | pending_after_get={_tts_q.qsize()} | text={text[:60]}"
        )
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

    asr = StreamingASR()
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

                # Queue TTS requests so playback stays serial and does not interrupt itself.
                _tts_q.put((translation, MT_TARGET_LANG))
                print(
                    f"[TTS dispatch] queued | lang={MT_TARGET_LANG} | queue_size={_tts_q.qsize()} | text={translation[:60]}"
                )

    except KeyboardInterrupt:
        print("\nPipeline stopped.")


if __name__ == "__main__":
    main()
