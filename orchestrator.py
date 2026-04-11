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
import json
import os
import queue
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass
from itertools import count

import numpy as np
import requests
import sounddevice as sd
import torch
from dotenv import load_dotenv
from pyrnnoise import RNNoise

from app_paths import runtime_dir, runtime_path

load_dotenv(runtime_path(".env"), override=True)

from app_logging import configure_logging, default_log_path, get_logger
from asr.aec import EchoCanceller
from asr.hotword_manager import HotwordManager
from asr.speaker_matcher import SpeakerDecision, SpeakerMatcher
from mt.prompt_context import MTPromptContext
from mt.scene_analyzer import SceneAnalysis, parse_scene_analysis

configure_logging()

bootstrap_log = get_logger("BOOT")
bootstrap_log.info("startup | importing TTS module")

from main import speak as tts_speak

bootstrap_log.info("startup | TTS module imported")


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _normalize_asr_mode(value: str) -> str:
    normalized = (value or "").strip().lower()
    aliases = {
        "api_only": "api_only",
        "api-only": "api_only",
        "api": "api_only",
        "remote_only": "api_only",
        "api_local_fallback": "api_local_fallback",
        "api-local-fallback": "api_local_fallback",
        "api+local": "api_local_fallback",
        "api+local_fallback": "api_local_fallback",
        "fallback": "api_local_fallback",
    }
    return aliases.get(normalized, "api_only")


GLOBAL_API_ONLY = _env_bool("API_ONLY", False)
USE_LOCAL_MT = False if GLOBAL_API_ONLY else _env_bool("USE_LOCAL_MT", False)
USE_QWEN_ASR_API = os.getenv("USE_QWEN_ASR_API", "false").lower() == "true"
ASR_MODE = "api_only" if GLOBAL_API_ONLY else _normalize_asr_mode(os.getenv("ASR_MODE", "api_only"))
API_ONLY_ASR = ASR_MODE == "api_only"

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

BASE_DIR = str(runtime_dir())
MODEL_PATH = os.path.join(BASE_DIR, "models", "zipformer")
TORCH_HUB_DIR = os.getenv("TORCH_HUB_DIR", os.path.join(BASE_DIR, "models", "torch_hub"))
SILERO_VAD_REPO_DIR = os.path.join(TORCH_HUB_DIR, "snakers4_silero-vad_master")

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
MT_SCENE_ANALYZER_ENABLED = os.getenv("MT_SCENE_ANALYZER_ENABLED", "true").lower() == "true"
MT_SCENE_ANALYZER_MODEL = os.getenv("MT_SCENE_ANALYZER_MODEL", DEEPSEEK_MODEL)
MT_SCENE_ANALYZER_TIMEOUT_SEC = int(os.getenv("MT_SCENE_ANALYZER_TIMEOUT_SEC", str(MT_TIMEOUT_SEC)))
MT_SCENE_ANALYZER_REFRESH_TURNS = max(1, int(os.getenv("MT_SCENE_ANALYZER_REFRESH_TURNS", "10")))
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
SILERO_VAD_ENABLED = os.getenv("SILERO_VAD_ENABLED", "true").lower() == "true"

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
latency_log = get_logger("LATENCY")
audio_log = get_logger("Audio")
queue_log = get_logger("Queue")
metrics_log = get_logger("ASR.METRICS")
hotword_log = get_logger("ASR.HOTWORD")
mt_context_log = get_logger("MT.CONTEXT")
speaker_log = get_logger("SPEAKER")
turn_log = get_logger("TURN")


_trace_counter = count(1)


def _now() -> float:
    return time.perf_counter()


def _elapsed_ms(start: float, end: float) -> float:
    if start <= 0.0 or end <= 0.0 or end < start:
        return 0.0
    return (end - start) * 1000.0


@dataclass
class _PipelineTrace:
    trace_id: int
    asr_started_at: float = 0.0
    asr_speech_end_at: float = 0.0
    asr_first_partial_at: float = 0.0
    asr_last_partial_at: float = 0.0
    asr_partial_count: int = 0
    asr_final_at: float = 0.0
    mt_scene_started_at: float = 0.0
    mt_scene_done_at: float = 0.0
    mt_scene_cache_hit: bool = False
    mt_translate_started_at: float = 0.0
    mt_translate_done_at: float = 0.0
    tts_started_at: float = 0.0
    tts_ready_at: float = 0.0
    tts_done_at: float = 0.0


@dataclass
class _ASRTextEvent:
    text: str
    trace: _PipelineTrace
    speaker_id: str | None = None
    voice_id: str | None = None
    speaker_registry_label: str | None = None
    speaker_registry_sample_alias: str | None = None
    speaker_session_score: float = 0.0
    speaker_registry_score: float = 0.0
    speaker_registry_margin: float = 0.0
    speaker_best_registry_label: str | None = None
    speaker_best_registry_score: float = 0.0
    speaker_second_registry_label: str | None = None
    speaker_second_registry_score: float = 0.0


@dataclass
class _TTSEvent:
    text: str
    lang: str
    trace: _PipelineTrace
    speaker_id: str | None = None
    voice_id: str | None = None


def _new_trace() -> _PipelineTrace:
    return _PipelineTrace(trace_id=next(_trace_counter))


def _log_trace_latency(trace: _PipelineTrace) -> None:
    latency_log.info(
        "trace | id=%s | asr_latency_ms=%.1f | asr_first_partial_ms=%.1f | asr_final_tail_ms=%.1f | "
        "mt_scene_analyzer_latency_ms=%.1f | mt_translator_latency_ms=%.1f | tts_latency_ms=%.1f | "
        "end_to_end_latency_ms=%.1f | scene_cache_hit=%s | partials=%s",
        trace.trace_id,
        _elapsed_ms(trace.asr_started_at, trace.asr_speech_end_at or trace.asr_final_at),
        _elapsed_ms(trace.asr_started_at, trace.asr_first_partial_at),
        _elapsed_ms(trace.asr_speech_end_at, trace.asr_final_at),
        _elapsed_ms(trace.mt_scene_started_at, trace.mt_scene_done_at),
        _elapsed_ms(trace.mt_translate_started_at, trace.mt_translate_done_at),
        _elapsed_ms(trace.tts_started_at, trace.tts_ready_at),
        _elapsed_ms(trace.asr_started_at, trace.tts_ready_at),
        trace.mt_scene_cache_hit,
        trace.asr_partial_count,
    )


def _clip_text(text: str, limit: int = 72) -> str:
    cleaned = (text or "").replace("\r", " ").replace("\n", " ").strip()
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[: max(0, limit - 1)] + "..."


def _log_turn_summary(event: _ASRTextEvent, translation: str) -> None:
    turn_log.info(
        "turn | id=%s | speaker=%s | route=%s | voice=%s | top1=%s:%.3f | top2=%s:%.3f | asr=%s | mt=%s",
        event.trace.trace_id,
        event.speaker_id or "unknown",
        event.speaker_registry_sample_alias or event.speaker_registry_label or "none",
        event.voice_id or "default",
        event.speaker_best_registry_label or "none",
        event.speaker_best_registry_score,
        event.speaker_second_registry_label or "none",
        event.speaker_second_registry_score,
        _clip_text(event.text),
        _clip_text(translation),
    )


def _observe_asr_partial(trace: _PipelineTrace, text: str) -> None:
    if not text:
        return
    now = _now()
    trace.asr_last_partial_at = now
    trace.asr_partial_count += 1
    if trace.asr_first_partial_at <= 0.0:
        trace.asr_first_partial_at = now
    asr_log.info("partial | text=%s", text)


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
_mt_scene_cache = {
    "analysis": SceneAnalysis(),
    "turns_since_refresh": MT_SCENE_ANALYZER_REFRESH_TURNS,
    "refresh_in_flight": False,
}
_mt_scene_cache_lock = threading.Lock()


def _postprocess_asr_final(text: str) -> str:
    rewritten, hits = _hotword_manager.rewrite(text)
    if not hits or rewritten == text:
        return text

    _asr_metrics.observe_hotword_rewrite()
    hits_desc = ", ".join(f"{alias}->{canonical}" for alias, canonical in hits)
    hotword_log.info("matches=%s | original=%s | rewritten=%s", hits_desc, text, rewritten)
    return rewritten


def _mt_context_prompt(text: str, trace: _PipelineTrace) -> str:
    analysis = _analyze_mt_scene(text, trace)
    prompt = _mt_prompt_context.build_translation_prompt(analysis.summary_block())
    if prompt:
        mt_context_log.info(
            "scene=%s | utterance_type=%s | entity_focus=%s | confidence=%s",
            analysis.scene,
            analysis.utterance_type,
            analysis.entity_focus,
            analysis.confidence,
        )
    return prompt


def _mt_remember_turn(text: str, translated: str) -> None:
    _mt_prompt_context.observe_turn(text, translated)


def _call_deepseek_chat(
    *,
    messages: list[dict[str, str]],
    model: str,
    timeout_sec: int,
    stage: str,
) -> str:
    mt_context_log.info(
        "deepseek_call | stage=%s | model=%s | timeout=%ss",
        stage,
        model,
        timeout_sec,
    )
    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0.0,
    }

    resp = requests.post(
        f"{DEEPSEEK_BASE_URL}/chat/completions",
        headers=headers,
        json=payload,
        timeout=timeout_sec,
    )
    resp.raise_for_status()
    data = resp.json()
    content = data["choices"][0]["message"]["content"].strip()
    mt_context_log.info(
        "deepseek_done | stage=%s | chars=%s",
        stage,
        len(content),
    )
    return content


def _build_scene_analyzer_messages(text: str) -> list[dict[str, str]]:
    recent_context = _mt_prompt_context.recent_context_block()
    context_block = f"\n\n{recent_context}" if recent_context else ""
    return [
        {
            "role": "system",
            "content": (
                "You are a scene disambiguation expert for a real-time speech translation system. "
                "Analyze the current utterance and return only compact JSON. "
                "Use this schema exactly: "
                '{"scene":"general|campus|technical|literature|daily_chat|other",'
                '"utterance_type":"statement|question|fragment|exclamation|other",'
                '"entity_focus":"person|organization|course|concept|none",'
                '"register":"spoken|formal|mixed",'
                '"translation_hint":"short guidance for the translator",'
                '"confidence":"high|medium|low"}'
            ),
        },
        {
            "role": "user",
            "content": (
                f"Source language: {MT_SOURCE_LANG}\n"
                f"Target language: {MT_TARGET_LANG}\n"
                f"Current utterance: {text}"
                f"{context_block}"
            ),
        },
    ]


def _refresh_mt_scene_async(text: str) -> None:
    try:
        messages = _build_scene_analyzer_messages(text)
        response_text = _call_deepseek_chat(
            messages=messages,
            model=MT_SCENE_ANALYZER_MODEL,
            timeout_sec=MT_SCENE_ANALYZER_TIMEOUT_SEC,
            stage="scene_analyzer",
        )
        analysis = parse_scene_analysis(response_text)
        mt_context_log.info(
            "scene_result | scene=%s | utterance_type=%s | entity_focus=%s | register=%s | confidence=%s",
            analysis.scene,
            analysis.utterance_type,
            analysis.entity_focus,
            analysis.register,
            analysis.confidence,
        )
        with _mt_scene_cache_lock:
            previous = _mt_scene_cache["analysis"]
            if analysis.cache_key() == previous.cache_key():
                mt_context_log.info("scene_cache_refresh | changed=False")
            else:
                mt_context_log.info(
                    "scene_cache_refresh | changed=True | old_scene=%s | new_scene=%s",
                    previous.scene,
                    analysis.scene,
                )
                _mt_scene_cache["analysis"] = analysis
            _mt_scene_cache["turns_since_refresh"] = 0
    except requests.exceptions.Timeout:
        mt_context_log.warning("scene analyzer timed out; keeping previous cached analysis")
    except Exception as exc:
        mt_context_log.warning("scene analyzer failed: %s", exc)
    finally:
        with _mt_scene_cache_lock:
            _mt_scene_cache["refresh_in_flight"] = False


def _start_mt_scene_refresh(text: str) -> None:
    thread = threading.Thread(
        target=_refresh_mt_scene_async,
        args=(text,),
        daemon=True,
        name="mt-scene-refresh",
    )
    thread.start()


def _analyze_mt_scene(text: str, trace: _PipelineTrace) -> SceneAnalysis:
    if not MT_SCENE_ANALYZER_ENABLED:
        return SceneAnalysis()

    should_refresh = False
    with _mt_scene_cache_lock:
        cached_analysis = _mt_scene_cache["analysis"]
        turns_since_refresh = int(_mt_scene_cache["turns_since_refresh"])
        refresh_in_flight = bool(_mt_scene_cache["refresh_in_flight"])

        if turns_since_refresh < MT_SCENE_ANALYZER_REFRESH_TURNS:
            _mt_scene_cache["turns_since_refresh"] = turns_since_refresh + 1
            trace.mt_scene_cache_hit = True
            mt_context_log.info(
                "scene_cache_hit | turn=%s/%s | scene=%s | entity_focus=%s",
                turns_since_refresh + 1,
                MT_SCENE_ANALYZER_REFRESH_TURNS,
                cached_analysis.scene,
                cached_analysis.entity_focus,
            )
            return cached_analysis

        trace.mt_scene_cache_hit = True
        if refresh_in_flight:
            mt_context_log.info(
                "scene_cache_stale | reason=refresh_in_flight | scene=%s | entity_focus=%s",
                cached_analysis.scene,
                cached_analysis.entity_focus,
            )
        else:
            _mt_scene_cache["refresh_in_flight"] = True
            _mt_scene_cache["turns_since_refresh"] = MT_SCENE_ANALYZER_REFRESH_TURNS + 1
            should_refresh = True
            mt_context_log.info(
                "scene_cache_refresh_scheduled | scene=%s | entity_focus=%s | refresh_turns=%s",
                cached_analysis.scene,
                cached_analysis.entity_focus,
                MT_SCENE_ANALYZER_REFRESH_TURNS,
            )

    if should_refresh:
        _start_mt_scene_refresh(text)

    return cached_analysis


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
    utterance_audio: np.ndarray | None = None


class _SpeechFrontEnd:
    def __init__(self):
        self.echo_canceller = EchoCanceller(frame_size=FRAME_SIZE, sample_rate=SAMPLE_RATE)
        get_logger("AEC").info(self.echo_canceller.describe())
        self.rnnoise = RNNoise(sample_rate=SAMPLE_RATE)
        self.vad_model = None
        self._silero_available = False
        self._init_vad_model()
        self._pre_roll: deque[np.ndarray] = deque(maxlen=PRE_SPEECH_FRAMES)
        self._vad_history: deque[float] = deque(maxlen=VAD_SMOOTHING_FRAMES)
        self._in_speech = False
        self._speech_frames = 0
        self._start_frames = 0
        self._silence_frames = 0
        self._hpf_prev_x = 0.0
        self._hpf_prev_y = 0.0
        self._noise_floor = ENERGY_THRESHOLD
        self._utterance_audio_chunks: list[np.ndarray] = []

    def _init_vad_model(self) -> None:
        if not SILERO_VAD_ENABLED:
            asr_log.warning("Silero VAD disabled by config; frontend will use energy-gated fallback.")
            return
        try:
            os.makedirs(TORCH_HUB_DIR, exist_ok=True)
            torch.hub.set_dir(TORCH_HUB_DIR)
        except Exception as exc:
            asr_log.warning("Failed to prepare torch hub dir '%s': %s", TORCH_HUB_DIR, exc)

        try:
            if os.path.isdir(SILERO_VAD_REPO_DIR):
                vad_model, _ = torch.hub.load(
                    repo_or_dir=SILERO_VAD_REPO_DIR,
                    model="silero_vad",
                    source="local",
                    force_reload=False,
                    verbose=False,
                )
                asr_log.info("Silero VAD loaded from local cache: %s", SILERO_VAD_REPO_DIR)
            else:
                vad_model, _ = torch.hub.load(
                    repo_or_dir="snakers4/silero-vad",
                    model="silero_vad",
                    force_reload=False,
                    verbose=False,
                    trust_repo=True,
                )
                asr_log.info("Silero VAD downloaded via torch.hub into %s", TORCH_HUB_DIR)
            vad_model.eval()
            self.vad_model = vad_model
            self._silero_available = True
        except Exception as exc:
            self.vad_model = None
            self._silero_available = False
            asr_log.warning(
                "Silero VAD unavailable, frontend will fall back to energy-gated VAD only: %s",
                exc,
            )

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

    def _vad_prob(self, audio: np.ndarray, rms: float | None = None) -> float:
        if self.vad_model is None:
            if rms is None:
                rms = float(np.sqrt(np.mean(np.square(audio), dtype=np.float32)))
            baseline = max(ENERGY_THRESHOLD, 1e-4)
            return float(max(0.0, min(1.0, rms / baseline)))
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
        self._utterance_audio_chunks = []

    def process_audio(self, audio: np.ndarray) -> _FrontEndDecision:
        cleaned = self.echo_canceller.process_capture(audio)
        filtered = self._high_pass(cleaned)
        denoised = self._denoise(filtered)
        if denoised.size == 0:
            return _FrontEndDecision(feed_chunks=[])

        rms = float(np.sqrt(np.mean(np.square(denoised), dtype=np.float32)))
        vad_prob = self._vad_prob(denoised, rms=rms)
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
            self._utterance_audio_chunks = [chunk.copy() for chunk in feed_chunks]
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
        self._utterance_audio_chunks.append(denoised.copy())
        if continue_gate:
            self._silence_frames = 0
        else:
            self._silence_frames += 1

        should_finalize = self._silence_frames >= MAX_SILENCE_FRAMES
        speech_frames = self._speech_frames
        utterance_audio = None
        if should_finalize:
            if self._utterance_audio_chunks:
                utterance_audio = np.concatenate(self._utterance_audio_chunks, dtype=np.float32)
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
            utterance_audio=utterance_audio,
        )

_speaker_matcher = SpeakerMatcher.from_env(base_dir=BASE_DIR, sample_rate=SAMPLE_RATE)


class LocalStreamingASR:
    def __init__(self):
        try:
            import sherpa_onnx
        except ImportError as exc:
            raise ImportError(
                "Local ASR fallback requires sherpa_onnx. "
                "Install local ASR dependencies or switch ASR_MODE=api_only."
            ) from exc

        tokens_path = os.path.join(MODEL_PATH, "tokens.txt")
        encoder_path = os.path.join(MODEL_PATH, "encoder-epoch-99-avg-1.int8.onnx")
        decoder_path = os.path.join(MODEL_PATH, "decoder-epoch-99-avg-1.onnx")
        joiner_path = os.path.join(MODEL_PATH, "joiner-epoch-99-avg-1.int8.onnx")

        asr_log.info(
            "model | provider=sherpa-onnx | model_path=%s | decoding_method=greedy_search",
            MODEL_PATH,
        )
        asr_log.info(
            "vad | silero_enabled=%s | torch_hub_dir=%s | start_threshold=%.2f | "
            "end_threshold=%.2f | energy_threshold=%.4f | adaptive_energy=%s | max_silence_frames=%s",
            SILERO_VAD_ENABLED,
            TORCH_HUB_DIR,
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
        self.last_partial_text = ""
        self._active_trace: _PipelineTrace | None = None

    def _reset(self):
        self.stream = self.recognizer.create_stream()
        self.last_text = ""
        self.last_partial_text = ""
        self._active_trace = None

    def process_audio(self, audio: np.ndarray) -> _ASRTextEvent | None:
        decision = self.frontend.process_audio(audio)
        _asr_metrics.observe_frontend(decision)

        if decision.speech_started and self._active_trace is None:
            self._active_trace = _new_trace()
            self._active_trace.asr_started_at = _now()

        for chunk in decision.feed_chunks:
            if self._active_trace is None:
                self._active_trace = _new_trace()
                self._active_trace.asr_started_at = _now()
            self.stream.accept_waveform(SAMPLE_RATE, chunk)
            while self.recognizer.is_ready(self.stream):
                self.recognizer.decode_stream(self.stream)

        if decision.feed_chunks:
            current = self.recognizer.get_result(self.stream).strip()
            if current and current != self.last_text:
                self.last_text = current
            if current and current != self.last_partial_text and self._active_trace is not None:
                self.last_partial_text = current
                _observe_asr_partial(self._active_trace, current)

        if decision.should_finalize:
            if self._active_trace is not None and self._active_trace.asr_speech_end_at <= 0.0:
                self._active_trace.asr_speech_end_at = _now()
            final = self.recognizer.get_result(self.stream).strip() or self.last_text
            trace = self._active_trace or _new_trace()
            trace.asr_final_at = _now()
            speaker_decision = _speaker_matcher.match_utterance(decision.utterance_audio)
            self._reset()
            if decision.should_reset:
                return None
            if final:
                final = _postprocess_asr_final(final)
                _asr_metrics.observe_final(final)
                asr_log.info("final | text=%s", final)
                return _ASRTextEvent(
                    text=final,
                    trace=trace,
                    speaker_id=speaker_decision.speaker_id if speaker_decision else None,
                    voice_id=speaker_decision.voice_id if speaker_decision else None,
                    speaker_registry_label=(
                        speaker_decision.registry_label if speaker_decision else None
                    ),
                    speaker_registry_sample_alias=(
                        speaker_decision.registry_sample_alias if speaker_decision else None
                    ),
                    speaker_session_score=(
                        speaker_decision.session_score if speaker_decision else 0.0
                    ),
                    speaker_registry_score=(
                        speaker_decision.registry_score if speaker_decision else 0.0
                    ),
                    speaker_registry_margin=(
                        speaker_decision.registry_margin if speaker_decision else 0.0
                    ),
                    speaker_best_registry_label=(
                        speaker_decision.best_registry_label if speaker_decision else None
                    ),
                    speaker_best_registry_score=(
                        speaker_decision.best_registry_score if speaker_decision else 0.0
                    ),
                    speaker_second_registry_label=(
                        speaker_decision.second_registry_label if speaker_decision else None
                    ),
                    speaker_second_registry_score=(
                        speaker_decision.second_registry_score if speaker_decision else 0.0
                    ),
                )

        return None

    def close(self) -> None:
        return None


class _QwenASRCallback:
    def __init__(self, final_queue: queue.Queue[str], partial_queue: queue.Queue[str]):
        self.final_queue = final_queue
        self.partial_queue = partial_queue
        self._partial_debug_logged = False
        self._event_debug_types_logged: set[str] = set()

    def on_open(self):
        return None

    def on_close(self, code, msg):
        return None

    def on_event(self, response):
        try:
            event_type = response.get("type", "")
            if (
                event_type
                and event_type not in self._event_debug_types_logged
                and (
                    "transcript" in event_type
                    or "transcription" in event_type
                    or "audio" in event_type
                    or "conversation.item" in event_type
                )
            ):
                self._event_debug_types_logged.add(event_type)
                asr_log.info("qwen_event_debug | type=%s | payload=%s", event_type, response)
            if event_type == "response.audio_transcript.delta":
                if not self._partial_debug_logged:
                    self._partial_debug_logged = True
                    asr_log.info("partial_debug | payload=%s", response)
                partial = (response.get("delta") or response.get("transcript") or "").strip()
                if partial:
                    self.partial_queue.put(partial)
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
        self._partial_queue: queue.Queue[str] = queue.Queue()
        self._callback = _QwenASRCallback(self._final_queue, self._partial_queue)
        self.frontend = _SpeechFrontEnd()
        self._conversation = None
        self._active_trace: _PipelineTrace | None = None
        self._pending_traces: deque[_PipelineTrace] = deque()
        self._pending_speakers: deque[SpeakerDecision | None] = deque()
        self._deferred_finals: deque[str] = deque()
        self._last_trace_candidate: _PipelineTrace | None = None
        self._last_partial_text = ""
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

    def process_audio(self, audio: np.ndarray) -> _ASRTextEvent | None:
        decision = self.frontend.process_audio(audio)
        _asr_metrics.observe_frontend(decision)

        if decision.speech_started and self._active_trace is None:
            self._active_trace = _new_trace()
            self._active_trace.asr_started_at = _now()
            self._last_trace_candidate = self._active_trace

        for chunk in decision.feed_chunks:
            if self._active_trace is None:
                self._active_trace = _new_trace()
                self._active_trace.asr_started_at = _now()
            self._last_trace_candidate = self._active_trace
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
        if decision.should_finalize and self._active_trace is not None:
            if self._active_trace.asr_speech_end_at <= 0.0:
                self._active_trace.asr_speech_end_at = _now()
            if not decision.should_reset:
                self._pending_traces.append(self._active_trace)
                self._pending_speakers.append(
                    _speaker_matcher.match_utterance(decision.utterance_audio)
                )
                self._last_trace_candidate = self._active_trace
            self._active_trace = None
        current_trace = self._active_trace or (self._pending_traces[-1] if self._pending_traces else self._last_trace_candidate)
        while True:
            try:
                partial = self._partial_queue.get_nowait()
            except queue.Empty:
                break
            if partial and partial != self._last_partial_text and current_trace is not None:
                self._last_partial_text = partial
                _observe_asr_partial(current_trace, partial)
        while True:
            try:
                final = self._final_queue.get_nowait()
            except queue.Empty:
                break
            if final:
                self._deferred_finals.append(final)

        if not self._deferred_finals:
            return None

        if self._pending_traces:
            final = self._deferred_finals.popleft()
            if self._pending_traces:
                trace = self._pending_traces.popleft()
                speaker_decision = self._pending_speakers.popleft() if self._pending_speakers else None
        elif self._active_trace is not None:
            # The remote final transcript arrived before local VAD finalized the utterance.
            # Hold it until we have the matching speaker decision.
            return None
        elif self._last_trace_candidate is not None:
            final = self._deferred_finals.popleft()
            trace = self._last_trace_candidate
            speaker_decision = None
        else:
            final = self._deferred_finals.popleft()
            trace = _new_trace()
            trace.asr_started_at = _now()
            speaker_decision = None

        if trace.asr_speech_end_at <= 0.0:
            trace.asr_speech_end_at = _now()
        trace.asr_final_at = _now()
        self._last_trace_candidate = trace
        self._last_partial_text = ""
        final = _postprocess_asr_final(final)
        _asr_metrics.observe_final(final)
        asr_log.info("final | text=%s", final)
        return _ASRTextEvent(
            text=final,
            trace=trace,
            speaker_id=speaker_decision.speaker_id if speaker_decision else None,
            voice_id=speaker_decision.voice_id if speaker_decision else None,
            speaker_registry_label=(
                speaker_decision.registry_label if speaker_decision else None
            ),
            speaker_registry_sample_alias=(
                speaker_decision.registry_sample_alias if speaker_decision else None
            ),
            speaker_session_score=(
                speaker_decision.session_score if speaker_decision else 0.0
            ),
            speaker_registry_score=(
                speaker_decision.registry_score if speaker_decision else 0.0
            ),
            speaker_registry_margin=(
                speaker_decision.registry_margin if speaker_decision else 0.0
            ),
            speaker_best_registry_label=(
                speaker_decision.best_registry_label if speaker_decision else None
            ),
            speaker_best_registry_score=(
                speaker_decision.best_registry_score if speaker_decision else 0.0
            ),
            speaker_second_registry_label=(
                speaker_decision.second_registry_label if speaker_decision else None
            ),
            speaker_second_registry_score=(
                speaker_decision.second_registry_score if speaker_decision else 0.0
            ),
        )

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
            asr_log.info(
                "route | mode=%s | primary=qwen_api | fallback=%s",
                ASR_MODE,
                "zipformer" if not API_ONLY_ASR else "none",
            )
            return asr
        except Exception as exc:
            if API_ONLY_ASR:
                raise RuntimeError(
                    f"Qwen ASR unavailable and API-only mode is enabled: {exc}"
                ) from exc
            asr_log.warning("Qwen ASR unavailable, fallback to local zipformer: %s", exc)
    elif API_ONLY_ASR:
        raise RuntimeError("ASR_MODE=api_only requires USE_QWEN_ASR_API=true")
    asr = LocalStreamingASR()
    asr_log.info("route | mode=%s | primary=zipformer", ASR_MODE)
    return asr


def _translate_local(text: str, trace: _PipelineTrace) -> str | None:
    try:
        trace.mt_translate_started_at = _now()
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
        trace.mt_translate_done_at = _now()
        if result:
            _mt_remember_turn(text, result)
            mt_log.info("result | text=%s", result)
        return result or None
    except requests.exceptions.Timeout:
        trace.mt_translate_done_at = _now()
        mt_log.warning("Timed out - skipping utterance.")
        return None
    except Exception as exc:
        trace.mt_translate_done_at = _now()
        mt_log.error("Error: %s", exc)
        return None


def _translate_deepseek(text: str, trace: _PipelineTrace) -> str | None:
    context_prompt = _mt_context_prompt(text, trace)
    prompt_suffix = ""
    if context_prompt:
        prompt_suffix = "\n\n" + context_prompt
    try:
        source_name = LANGUAGE_NAME_MAP.get(MT_SOURCE_LANG, MT_SOURCE_LANG)
        target_name = LANGUAGE_NAME_MAP.get(MT_TARGET_LANG, MT_TARGET_LANG)
        messages = [
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
                    "Use the provided scene analysis only for disambiguation. "
                    "Keep proper nouns and domain terms consistent across turns. "
                    f"{prompt_suffix}"
                ),
            },
            {
                "role": "user",
                "content": text,
            },
        ]

        trace.mt_translate_started_at = _now()
        result = _call_deepseek_chat(
            messages=messages,
            model=DEEPSEEK_MODEL,
            timeout_sec=MT_TIMEOUT_SEC,
            stage="translator",
        )
        trace.mt_translate_done_at = _now()
        if result:
            _mt_remember_turn(text, result)
            mt_log.info("result | text=%s", result)
        return result or None
    except requests.exceptions.Timeout:
        trace.mt_translate_done_at = _now()
        mt_log.warning("Timed out - skipping utterance.")
        return None
    except Exception as exc:
        trace.mt_translate_done_at = _now()
        mt_log.error("Exception: %s", exc)
        return None


def translate(text: str, trace: _PipelineTrace) -> str | None:
    if USE_LOCAL_MT:
        return _translate_local(text, trace)
    return _translate_deepseek(text, trace)


_audio_q: queue.Queue[np.ndarray] = queue.Queue(maxsize=AUDIO_QUEUE_MAX_CHUNKS)
_asr_text_q: queue.Queue[_ASRTextEvent] = queue.Queue(maxsize=ASR_RESULT_QUEUE_MAX)
_tts_q: queue.Queue[_TTSEvent] = queue.Queue(maxsize=TTS_QUEUE_MAX)
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


def _run_tts(event: _TTSEvent) -> None:
    try:
        event.trace.tts_started_at = _now()
        result = tts_speak(event.text, lang=event.lang, voice=event.voice_id)
        event.trace.tts_done_at = _now()
        if result is not None:
            event.trace.tts_ready_at = event.trace.tts_started_at + (result.provider_ready_ms / 1000.0)
        else:
            event.trace.tts_ready_at = event.trace.tts_done_at
        _log_trace_latency(event.trace)
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
                final_event = asr.process_audio(chunk)
                if final_event:
                    _put_latest(_asr_text_q, final_event, "asr_text")
            except Exception as exc:
                _asr_metrics.observe_asr_error()
                asr_log.warning("worker recoverable error: %s", exc)
                time.sleep(0.2)
        finally:
            _audio_q.task_done()


def _mt_worker(stop_event: threading.Event) -> None:
    while not stop_event.is_set():
        try:
            event = _asr_text_q.get(timeout=0.2)
        except queue.Empty:
            continue

        try:
            translation = translate(event.text, event.trace)
            if translation:
                _log_turn_summary(event, translation)
                if event.speaker_id:
                    speaker_log.info(
                        "speaker_route | session=%s -> registry=%s -> voice=%s | session_score=%.3f | registry_score=%.3f",
                        event.speaker_id,
                        event.speaker_registry_sample_alias or event.speaker_registry_label or "none",
                        event.voice_id or "default",
                        event.speaker_session_score,
                        event.speaker_registry_score,
                    )
                _put_latest(
                    _tts_q,
                    _TTSEvent(
                        text=translation,
                        lang=MT_TARGET_LANG,
                        trace=event.trace,
                        speaker_id=event.speaker_id,
                        voice_id=event.voice_id,
                    ),
                    "tts",
                )
        finally:
            _asr_text_q.task_done()


def _tts_worker(stop_event: threading.Event) -> None:
    while not stop_event.is_set():
        try:
            event = _tts_q.get(timeout=0.2)
        except queue.Empty:
            continue

        try:
            _run_tts(event)
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
    mt_context_log.info(
        "scene_analyzer | enabled=%s | model=%s | timeout=%ss",
        MT_SCENE_ANALYZER_ENABLED,
        MT_SCENE_ANALYZER_MODEL,
        MT_SCENE_ANALYZER_TIMEOUT_SEC,
    )
    mt_context_log.info(
        "scene_cache | refresh_turns=%s",
        MT_SCENE_ANALYZER_REFRESH_TURNS,
    )

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
