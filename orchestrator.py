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

from app_logging import configure_logging, default_log_path, get_logger
from asr.aec import EchoCanceller
from asr.hotword_manager import HotwordManager
from mt.prompt_context import MTPromptContext

load_dotenv(override=True)
configure_logging()

bootstrap_log = get_logger("BOOT")
bootstrap_log.info("startup | importing TTS module")

from main import speak as tts_speak

bootstrap_log.info("startup | TTS module imported")


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
ASR_METRICS_ENABLED = os.getenv("ASR_METRICS_ENABLED", "true").lower() == "true"
ASR_METRICS_LOG_INTERVAL_SEC = int(os.getenv("ASR_METRICS_LOG_INTERVAL_SEC", "30"))

RNNOISE_FRAME_SIZE = 160
HPF_ALPHA = 0.97
VAD_START_THRESHOLD = 0.55
VAD_END_THRESHOLD = 0.35
MIN_SPEECH_START_FRAMES = 2
MIN_SPEECH_FRAMES = 6
ENERGY_THRESHOLD = 0.008
ENERGY_RELEASE_RATIO = 0.65
ADAPTIVE_ENERGY_ENABLED = os.getenv("ADAPTIVE_ENERGY_ENABLED", "true").lower() == "true"
ADAPTIVE_ENERGY_NOISE_FLOOR_ALPHA = float(os.getenv("ADAPTIVE_ENERGY_NOISE_FLOOR_ALPHA", "0.02"))
ADAPTIVE_ENERGY_MIN_FACTOR = float(os.getenv("ADAPTIVE_ENERGY_MIN_FACTOR", "1.8"))
ADAPTIVE_ENERGY_MAX_FACTOR = float(os.getenv("ADAPTIVE_ENERGY_MAX_FACTOR", "3.0"))
ADAPTIVE_ENERGY_VAD_CEILING = float(os.getenv("ADAPTIVE_ENERGY_VAD_CEILING", "0.12"))
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

pipeline_log = get_logger("PIPELINE")
asr_log = get_logger("ASR")
mt_log = get_logger("MT")
audio_log = get_logger("Audio")
queue_log = get_logger("Queue")
metrics_log = get_logger("ASR.METRICS")
hotword_log = get_logger("ASR.HOTWORD")
mt_context_log = get_logger("MT.CONTEXT")


class _ASRMetrics:
    def __init__(self):
        self.enabled = ASR_METRICS_ENABLED
        self.log_interval_sec = max(5, ASR_METRICS_LOG_INTERVAL_SEC)
        self._lock = threading.Lock()
        self._started_at = time.monotonic()
        self._last_log_at = self._started_at
        self._reset_locked()

    def _reset_locked(self) -> None:
        self.frontend_frames = 0
        self.feed_decisions = 0
        self.feed_chunks = 0
        self.speech_start_count = 0
        self.finalize_count = 0
        self.short_reset_count = 0
        self.vad_prob_sum = 0.0
        self.rms_sum = 0.0
        self.dynamic_threshold_sum = 0.0
        self.noise_floor_sum = 0.0
        self.finalized_speech_frames = 0
        self.final_count = 0
        self.final_char_sum = 0
        self.hotword_rewrite_count = 0
        self.qwen_reconnect_count = 0
        self.asr_error_count = 0

    def observe_frontend(self, decision: "_FrontEndDecision") -> None:
        if not self.enabled:
            return
        with self._lock:
            self.frontend_frames += 1
            self.vad_prob_sum += float(decision.vad_prob)
            self.rms_sum += float(decision.rms)
            self.dynamic_threshold_sum += float(decision.dynamic_energy_threshold)
            self.noise_floor_sum += float(decision.noise_floor)
            if decision.feed_chunks:
                self.feed_decisions += 1
                self.feed_chunks += len(decision.feed_chunks)
            if decision.speech_started:
                self.speech_start_count += 1
            if decision.should_finalize:
                self.finalize_count += 1
                self.finalized_speech_frames += int(decision.speech_frames)
                if decision.should_reset:
                    self.short_reset_count += 1

    def observe_final(self, text: str) -> None:
        if not self.enabled:
            return
        stripped = text.strip()
        if not stripped:
            return
        with self._lock:
            self.final_count += 1
            self.final_char_sum += len(stripped)

    def observe_hotword_rewrite(self) -> None:
        if not self.enabled:
            return
        with self._lock:
            self.hotword_rewrite_count += 1

    def observe_qwen_reconnect(self) -> None:
        if not self.enabled:
            return
        with self._lock:
            self.qwen_reconnect_count += 1

    def observe_asr_error(self) -> None:
        if not self.enabled:
            return
        with self._lock:
            self.asr_error_count += 1

    def maybe_log(self, force: bool = False) -> None:
        if not self.enabled:
            return

        now = time.monotonic()
        with self._lock:
            elapsed = now - self._last_log_at
            if not force and elapsed < self.log_interval_sec:
                return

            snapshot = {
                "window_sec": elapsed,
                "frontend_frames": self.frontend_frames,
                "feed_decisions": self.feed_decisions,
                "feed_chunks": self.feed_chunks,
                "speech_start_count": self.speech_start_count,
                "finalize_count": self.finalize_count,
                "short_reset_count": self.short_reset_count,
                "vad_prob_sum": self.vad_prob_sum,
                "rms_sum": self.rms_sum,
                "dynamic_threshold_sum": self.dynamic_threshold_sum,
                "noise_floor_sum": self.noise_floor_sum,
                "finalized_speech_frames": self.finalized_speech_frames,
                "final_count": self.final_count,
                "final_char_sum": self.final_char_sum,
                "hotword_rewrite_count": self.hotword_rewrite_count,
                "qwen_reconnect_count": self.qwen_reconnect_count,
                "asr_error_count": self.asr_error_count,
            }
            self._reset_locked()
            self._last_log_at = now

        frontend_frames = max(1, snapshot["frontend_frames"])
        finalize_count = max(1, snapshot["finalize_count"])
        final_count = max(1, snapshot["final_count"])
        avg_vad = snapshot["vad_prob_sum"] / frontend_frames
        avg_rms = snapshot["rms_sum"] / frontend_frames
        avg_dynamic_threshold = snapshot["dynamic_threshold_sum"] / frontend_frames
        avg_noise_floor = snapshot["noise_floor_sum"] / frontend_frames
        avg_utt_ms = (
            snapshot["finalized_speech_frames"] * FRAME_SIZE * 1000.0 / SAMPLE_RATE / finalize_count
        )
        avg_final_chars = snapshot["final_char_sum"] / final_count if snapshot["final_count"] else 0.0

        metrics_log.info(
            "window=%.1fs | frontend_frames=%s | speech_starts=%s | finals=%s | "
            "finalizes=%s | short_resets=%s | feed_decisions=%s | feed_chunks=%s | "
            "avg_vad=%.3f | avg_rms=%.4f | avg_noise_floor=%.4f | avg_dynamic_threshold=%.4f | "
            "avg_utt_ms=%.0f | avg_final_chars=%.1f | hotword_rewrites=%s | "
            "audio_overflows=%s | audio_drops=%s | asr_text_drops=%s | tts_drops=%s | "
            "qwen_reconnects=%s | asr_errors=%s",
            snapshot["window_sec"],
            snapshot["frontend_frames"],
            snapshot["speech_start_count"],
            snapshot["final_count"],
            snapshot["finalize_count"],
            snapshot["short_reset_count"],
            snapshot["feed_decisions"],
            snapshot["feed_chunks"],
            avg_vad,
            avg_rms,
            avg_noise_floor,
            avg_dynamic_threshold,
            avg_utt_ms,
            avg_final_chars,
            snapshot["hotword_rewrite_count"],
            _audio_overflow_count,
            _audio_drop_count,
            _queue_drop_counts["asr_text"],
            _queue_drop_counts["tts"],
            snapshot["qwen_reconnect_count"],
            snapshot["asr_error_count"],
        )


_asr_metrics = _ASRMetrics()
_hotword_manager = HotwordManager()
_mt_prompt_context = MTPromptContext()


def _postprocess_asr_final(text: str) -> str:
    rewritten, hits = _hotword_manager.rewrite(text)
    if not hits or rewritten == text:
        return text

    _asr_metrics.observe_hotword_rewrite()
    hits_desc = ", ".join(f"{alias}->{canonical}" for alias, canonical in hits)
    hotword_log.info("matches=%s | original=%s | rewritten=%s", hits_desc, text, rewritten)
    return rewritten


def _mt_context_prompt(text: str) -> str:
    prompt = _mt_prompt_context.build_prompt(text, MT_SOURCE_LANG, MT_TARGET_LANG)
    if prompt:
        mt_context_log.info("prompt_built | text=%s", text)
    return prompt


def _mt_remember_turn(text: str, translated: str) -> None:
    _mt_prompt_context.observe_turn(text, translated)


@dataclass
class _FrontEndDecision:
    feed_chunks: list[np.ndarray]
    should_finalize: bool = False
    should_reset: bool = False
    speech_started: bool = False
    speech_frames: int = 0
    vad_prob: float = 0.0
    rms: float = 0.0
    noise_floor: float = ENERGY_THRESHOLD
    dynamic_energy_threshold: float = ENERGY_THRESHOLD


class _SpeechFrontEnd:
    def __init__(self):
        self.echo_canceller = EchoCanceller(frame_size=FRAME_SIZE, sample_rate=SAMPLE_RATE)
        get_logger("AEC").info(self.echo_canceller.describe())
        self.rnnoise = RNNoise(sample_rate=SAMPLE_RATE)
        self.vad_model, _ = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            force_reload=False,
            verbose=False,
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
        self._noise_floor = ENERGY_THRESHOLD

    def _update_noise_floor(self, rms: float, vad_prob: float) -> None:
        if not ADAPTIVE_ENERGY_ENABLED:
            return
        if self._in_speech:
            return
        if vad_prob > ADAPTIVE_ENERGY_VAD_CEILING:
            return

        alpha = min(max(ADAPTIVE_ENERGY_NOISE_FLOOR_ALPHA, 0.001), 0.2)
        self._noise_floor = ((1.0 - alpha) * self._noise_floor) + (alpha * rms)
        self._noise_floor = max(1e-5, self._noise_floor)

    def _energy_thresholds(self) -> tuple[float, float]:
        if not ADAPTIVE_ENERGY_ENABLED:
            start_threshold = ENERGY_THRESHOLD
        else:
            dynamic_floor = self._noise_floor * max(ADAPTIVE_ENERGY_MIN_FACTOR, 1.0)
            dynamic_ceiling = self._noise_floor * max(ADAPTIVE_ENERGY_MAX_FACTOR, ADAPTIVE_ENERGY_MIN_FACTOR)
            start_threshold = min(max(ENERGY_THRESHOLD, dynamic_floor), max(ENERGY_THRESHOLD, dynamic_ceiling))
        continue_threshold = start_threshold * ENERGY_RELEASE_RATIO
        return start_threshold, continue_threshold

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
        cleaned = self.echo_canceller.process_capture(audio)
        filtered = self._high_pass(cleaned)
        denoised = self._denoise(filtered)
        if denoised.size == 0:
            return _FrontEndDecision(feed_chunks=[])

        rms = float(np.sqrt(np.mean(np.square(denoised), dtype=np.float32)))
        vad_prob = self._vad_prob(denoised)
        self._vad_history.append(vad_prob)
        smoothed_prob = float(sum(self._vad_history) / len(self._vad_history))
        self._update_noise_floor(rms, smoothed_prob)
        energy_threshold, continue_energy_threshold = self._energy_thresholds()

        start_gate = smoothed_prob >= VAD_START_THRESHOLD and rms >= energy_threshold
        continue_gate = (
            smoothed_prob >= VAD_END_THRESHOLD
            and rms >= continue_energy_threshold
        )

        if not self._in_speech:
            self._pre_roll.append(denoised.copy())
            self._start_frames = self._start_frames + 1 if start_gate else 0
            if self._start_frames < MIN_SPEECH_START_FRAMES:
                return _FrontEndDecision(
                    feed_chunks=[],
                    vad_prob=smoothed_prob,
                    rms=rms,
                    noise_floor=self._noise_floor,
                    dynamic_energy_threshold=energy_threshold,
                )

            self._in_speech = True
            self._speech_frames = len(self._pre_roll)
            self._silence_frames = 0
            self._start_frames = 0
            feed_chunks = list(self._pre_roll)
            self._pre_roll.clear()
            return _FrontEndDecision(
                feed_chunks=feed_chunks,
                speech_started=True,
                speech_frames=self._speech_frames,
                vad_prob=smoothed_prob,
                rms=rms,
                noise_floor=self._noise_floor,
                dynamic_energy_threshold=energy_threshold,
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
            noise_floor=self._noise_floor,
            dynamic_energy_threshold=energy_threshold,
        )


class LocalStreamingASR:
    def __init__(self):
        tokens_path = os.path.join(MODEL_PATH, "tokens.txt")
        encoder_path = os.path.join(MODEL_PATH, "encoder-epoch-99-avg-1.int8.onnx")
        decoder_path = os.path.join(MODEL_PATH, "decoder-epoch-99-avg-1.onnx")
        joiner_path = os.path.join(MODEL_PATH, "joiner-epoch-99-avg-1.int8.onnx")

        asr_log.info(
            "model | provider=sherpa-onnx | model_path=%s | decoding_method=greedy_search",
            MODEL_PATH,
        )
        asr_log.info(
            "vad | repo=snakers4/silero-vad | model=silero_vad | start_threshold=%.2f | "
            "end_threshold=%.2f | energy_threshold=%.4f | adaptive_energy=%s | max_silence_frames=%s",
            VAD_START_THRESHOLD,
            VAD_END_THRESHOLD,
            ENERGY_THRESHOLD,
            ADAPTIVE_ENERGY_ENABLED,
            MAX_SILENCE_FRAMES,
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
        _asr_metrics.observe_frontend(decision)

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
                final = _postprocess_asr_final(final)
                _asr_metrics.observe_final(final)
                asr_log.info("final | text=%s", final)
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
            asr_log.warning("Qwen callback error: %s", exc)


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

        asr_log.info(
            "model | provider=qwen_api | model=%s | language=%s",
            QWEN_ASR_MODEL,
            QWEN_ASR_LANGUAGE,
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
        asr_log.warning("Qwen websocket closed, reconnecting...")
        _asr_metrics.observe_qwen_reconnect()
        try:
            if self._conversation is not None:
                self._conversation.close()
        except Exception:
            pass
        self._connect_conversation()

    def process_audio(self, audio: np.ndarray) -> str:
        decision = self.frontend.process_audio(audio)
        _asr_metrics.observe_frontend(decision)
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
            final = _postprocess_asr_final(final)
            _asr_metrics.observe_final(final)
            asr_log.info("final | text=%s", final)
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
            asr_log.info("route | primary=qwen_api | fallback=zipformer")
            return asr
        except Exception as exc:
            asr_log.warning("Qwen ASR unavailable, fallback to local zipformer: %s", exc)
    asr = LocalStreamingASR()
    asr_log.info("route | primary=zipformer")
    return asr


def _translate_local(text: str) -> str | None:
    try:
        mt_log.info(
            "request | url=%s | source=%s | target=%s | timeout=%ss",
            MT_URL,
            MT_SOURCE_LANG,
            MT_TARGET_LANG,
            MT_TIMEOUT_SEC,
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
            _mt_remember_turn(text, result)
            mt_log.info("result | text=%s", result)
        return result or None
    except requests.exceptions.Timeout:
        mt_log.warning("Timed out - skipping utterance.")
        return None
    except Exception as exc:
        mt_log.error("Error: %s", exc)
        return None


def _translate_deepseek(text: str) -> str | None:
    context_prompt = _mt_context_prompt(text)
    prompt_suffix = ""
    if context_prompt:
        prompt_suffix = "\n\n" + context_prompt
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
                        "If the input is a fragment, translate only that fragment. "
                        "Keep proper nouns and domain terms consistent across turns. "
                        f"{prompt_suffix}"
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
            _mt_remember_turn(text, result)
            mt_log.info("result | text=%s", result)
        return result or None
    except requests.exceptions.Timeout:
        mt_log.warning("Timed out - skipping utterance.")
        return None
    except Exception as exc:
        mt_log.error("Exception: %s", exc)
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
        queue_log.warning("%s queue full, dropping oldest item (drop_count=%s)", label, drop_count)


def _audio_callback(indata, frames, time_info, status):
    global _audio_overflow_count, _audio_drop_count
    if status:
        _audio_overflow_count += 1
        audio_log.warning("%s | overflow_count=%s", status, _audio_overflow_count)

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
            audio_log.warning(
                "capture queue full, dropping oldest buffered chunk (drop_count=%s)",
                _audio_drop_count,
            )


def _run_tts(text: str, lang: str) -> None:
    try:
        tts_speak(text, lang=lang)
    except Exception as exc:
        get_logger("TTS").error("thread crashed: %s", exc)


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
                _asr_metrics.observe_asr_error()
                asr_log.warning("worker recoverable error: %s", exc)
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
    pipeline_log.info("ASR -> MT -> TTS pipeline starting")
    pipeline_log.info(
        "config | audio_sample_rate=%s | frame_size=%s | channels=%s | input_latency=%s | "
        "audio_queue_max_chunks=%s | asr_result_queue_max=%s | tts_queue_max=%s | mt_target_lang=%s",
        SAMPLE_RATE,
        FRAME_SIZE,
        CHANNELS,
        AUDIO_INPUT_LATENCY,
        AUDIO_QUEUE_MAX_CHUNKS,
        ASR_RESULT_QUEUE_MAX,
        TTS_QUEUE_MAX,
        MT_TARGET_LANG,
    )
    pipeline_log.info("logs | file=%s", default_log_path())
    asr_log.info(
        "observability | enabled=%s | log_interval_sec=%s",
        ASR_METRICS_ENABLED,
        max(5, ASR_METRICS_LOG_INTERVAL_SEC),
    )
    hotword_log.info(_hotword_manager.describe())
    asr_log.info(
        "adaptive_energy | enabled=%s | base_threshold=%.4f | release_ratio=%.2f | "
        "noise_floor_alpha=%.3f | min_factor=%.2f | max_factor=%.2f | vad_ceiling=%.2f",
        ADAPTIVE_ENERGY_ENABLED,
        ENERGY_THRESHOLD,
        ENERGY_RELEASE_RATIO,
        ADAPTIVE_ENERGY_NOISE_FLOOR_ALPHA,
        ADAPTIVE_ENERGY_MIN_FACTOR,
        ADAPTIVE_ENERGY_MAX_FACTOR,
        ADAPTIVE_ENERGY_VAD_CEILING,
    )

    if USE_LOCAL_MT:
        mt_log.info(
            "model | provider=local_api | url=%s | source=%s | target=%s",
            MT_URL,
            MT_SOURCE_LANG,
            MT_TARGET_LANG,
        )
    else:
        mt_log.info("model | provider=deepseek | model=%s", DEEPSEEK_MODEL)
    mt_context_log.info(_mt_prompt_context.describe())

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

    pipeline_log.info("System ready - start speaking!")

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
                _asr_metrics.maybe_log()
    except KeyboardInterrupt:
        pipeline_log.info("Pipeline stopped.")
    finally:
        _asr_metrics.maybe_log(force=True)
        stop_event.set()
        for worker in workers:
            worker.join(timeout=1.0)
        asr.close()


if __name__ == "__main__":
    main()
