"""
main.py  -  Voice-cloning TTS engine (optimized for CPU speed)

Four optimizations applied over previous version
--------------------------------------------------
1. SPEAKER EMBEDDING CACHED AT STARTUP
   Previously: synthesize() called tts_to_file() via the high-level TTS wrapper,
   which re-reads the WAV and re-runs the speaker encoder on every single utterance.
   On CPU that encoder pass alone takes ~1-2 s.
   Fix: call model.get_conditioning_latents() once at init, cache the result, and
   pass the cached (gpt_cond_latent, speaker_embedding) directly to model.inference().

2. STREAMING INFERENCE WITH CHUNKED PLAYBACK
   Previously: full synthesis completed before playback began - the user heard
   nothing for the entire RTF*duration seconds, then audio played all at once.
   Fix: use model.inference_stream() to get audio chunks as they are generated,
   convert each chunk to int16, and push it into a sounddevice OutputStream.
   First audio now plays ~1-2 s after synthesis starts (first-token latency),
   regardless of total sentence length. Generation and playback overlap.

3. TORCH INFERENCE OPTIMISATIONS
   - torch.inference_mode() instead of torch.no_grad() (slightly faster, less memory)
   - torch.set_num_threads() set to all physical cores
   - enable_cudnn_benchmark kept off (not on GPU here) but added explicit
     float32 matmul precision hint which helps AVX2/AVX512 CPU paths

4. TEMPERATURE / SPEED TUNING
   - temperature=0.65 (was default 0.85) - fewer decoder missteps, less retrying
   - speed=1.1 - slightly faster speech, reduces audio duration and playback time
   - length_penalty=1.0, repetition_penalty=10.0 - prevents runaway generation
   These are conservative defaults; tune to taste in USER CONFIGURATION.
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import io
import json
import os
import queue
import sys
import tempfile
import threading
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

# Suppress the HuggingFace "deprecated generation config" warning from Coqui's
# stream_generator - it's cosmetic and not actionable in this app.
warnings.filterwarnings(
    "ignore",
    message="You have modified the pretrained model configuration to control generation",
    category=UserWarning,
)
warnings.filterwarnings(
    "ignore",
    message="pkg_resources is deprecated as an API.*",
    category=UserWarning,
)

import numpy as np
import redis
import torch
from dotenv import load_dotenv

load_dotenv(override=True)

from app_logging import configure_logging, get_logger
from asr.audio_player import play_audio_file
from asr.playback_bus import playback_bus

# =============================================================================
# USER CONFIGURATION
# =============================================================================

configure_logging()


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _normalize_tts_mode(value: str) -> str:
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

TTS_BACKEND = "xtts"                          # "xtts" | "openvoice" | "edge"
GLOBAL_API_ONLY = _env_bool("API_ONLY", False)
TTS_MODE = "api_only" if GLOBAL_API_ONLY else _normalize_tts_mode(os.getenv("TTS_MODE", "api_only"))
VOICE_SAMPLE = os.getenv(
    "VOICE_SAMPLE",
    "voice_samples/my_voice.wav",
)                                              # reference clip for cloning
PROXY = None                                  # e.g. "http://127.0.0.1:7890"
USE_QWEN_TTS_API = os.getenv("USE_QWEN_TTS_API", "false").lower() == "true"
API_ONLY_TTS = TTS_MODE == "api_only"

QWEN_TTS_MODEL = os.getenv("QWEN_TTS_MODEL", "qwen3-tts-vc-realtime-2026-01-15")
QWEN_TTS_VOICE = os.getenv("QWEN_TTS_VOICE", "").strip()
QWEN_TTS_URL = os.getenv(
    "QWEN_TTS_URL",
    "wss://dashscope.aliyuncs.com/api-ws/v1/realtime",
).rstrip("/")
QWEN_TTS_SESSION_MODE = os.getenv("QWEN_TTS_SESSION_MODE", "commit").strip().lower()
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY", "").strip()

# XTTS generation tuning (only used when TTS_BACKEND = "xtts")
XTTS_TEMPERATURE = 1.00
XTTS_SPEED = 1.0
XTTS_REPETITION_PENALTY = 10.0
XTTS_LENGTH_PENALTY = 1.0
XTTS_TOP_K = 50
XTTS_TOP_P = 0.85
XTTS_STREAM_CHUNK_SIZE = 20
XTTS_STREAMING_MODE = "auto"  # "auto" | "on" | "off"
XTTS_STREAM_PREROLL_CHUNKS = 4

EDGE_VOICE_MAP = {
    "zh": "zh-CN-XiaoxiaoNeural",
    "en": "en-US-AriaNeural",
    "ja": "ja-JP-NanamiNeural",
    "ko": "ko-KR-SunHiNeural",
    "fr": "fr-FR-DeniseNeural",
    "de": "de-DE-KatjaNeural",
}

RETRY_COUNT = 3
RETRY_DELAY = 1.0
QWEN_TTS_RESPONSE_TIMEOUT_SEC = float(os.getenv("QWEN_TTS_RESPONSE_TIMEOUT_SEC", "30"))
QWEN_TTS_SAMPLE_RATE = 24000
QWEN_TTS_STREAM_BLOCKSIZE = int(os.getenv("QWEN_TTS_STREAM_BLOCKSIZE", "2400"))

# =============================================================================

log = get_logger("TTS")


@dataclass(frozen=True)
class TTSTiming:
    provider: str
    mode: str
    provider_ready_ms: float
    total_ms: float
log.info("startup | configuring TTS backends")

_cpu_count = os.cpu_count() or 4
torch.set_num_threads(_cpu_count)
if hasattr(torch, "set_float32_matmul_precision"):
    torch.set_float32_matmul_precision("high")

_loop = asyncio.new_event_loop()
threading.Thread(target=_loop.run_forever, daemon=True, name="tts-loop").start()


# =============================================================================
# Compatibility patch: torch.isin() signature mismatch across PyTorch versions.
# =============================================================================

def _patch_torch_isin() -> None:
    current = torch.isin
    if getattr(current, "_codex_safe_isin", False):
        return

    def _safe_isin(
        elements,
        test_elements,
        *,
        assume_unique=False,
        invert=False,
        out=None,
    ):
        if not isinstance(test_elements, torch.Tensor):
            device = elements.device if isinstance(elements, torch.Tensor) else "cpu"
            dtype = elements.dtype if isinstance(elements, torch.Tensor) else torch.long
            if isinstance(test_elements, (list, tuple, set)):
                values = list(test_elements)
            else:
                values = [test_elements]
            test_elements = torch.tensor(values, dtype=dtype, device=device)

        kwargs = {
            "assume_unique": assume_unique,
            "invert": invert,
        }
        if out is not None:
            kwargs["out"] = out
        return current(elements, test_elements, **kwargs)

    _safe_isin._codex_safe_isin = True
    torch.isin = _safe_isin


def _patch_xtts_stream_modules() -> None:
    modules_to_patch = [
        "TTS.tts.layers.xtts.stream_generator",
        "TTS.tts.layers.xtts.tokenizer",
    ]
    for module_name in modules_to_patch:
        module = sys.modules.get(module_name)
        if module is None:
            continue
        module_torch = getattr(module, "torch", None)
        if module_torch is not None:
            module_torch.isin = torch.isin


_patch_torch_isin()


def _resolve_model_paths(manager, model_name: str) -> tuple[str, str]:
    captured_stdout = io.StringIO()
    captured_stderr = io.StringIO()
    with contextlib.redirect_stdout(captured_stdout), contextlib.redirect_stderr(captured_stderr):
        result = manager.download_model(model_name)

    for captured in (captured_stdout.getvalue(), captured_stderr.getvalue()):
        for line in captured.splitlines():
            line = line.strip()
            if line:
                log.info("XTTS setup | %s", line)

    model_path = None
    config_path = None

    if isinstance(result, (list, tuple)):
        if result:
            model_path = result[0]
        for item in result[1:]:
            if isinstance(item, (str, os.PathLike)) and str(item).endswith(".json"):
                config_path = str(item)
                break
    elif isinstance(result, (str, os.PathLike)):
        model_path = str(result)

    if model_path is None:
        raise RuntimeError(f"Could not resolve model path for '{model_name}'.")

    model_path = str(model_path)
    if config_path and Path(config_path).exists():
        return model_path, str(config_path)

    candidate_paths = [
        Path(model_path) / "config.json",
        Path(model_path) / model_name.replace("/", "--") / "config.json",
    ]

    for candidate in candidate_paths:
        if candidate.exists():
            return model_path, str(candidate)

    found = list(Path(model_path).rglob("config.json"))
    if found:
        return model_path, str(found[0])

    raise FileNotFoundError(
        f"Could not find config.json under XTTS model path: {model_path}"
    )


# =============================================================================
# Backend: Coqui XTTS-v2 (optimized)
# =============================================================================

class _XTTSBackend:
    SAMPLE_RATE = 24_000

    def __init__(self):
        try:
            from TTS.tts.configs.xtts_config import XttsConfig
            from TTS.tts.models.xtts import Xtts
            from TTS.utils.manage import ModelManager
            import TTS.tts.layers.xtts.stream_generator  # noqa: F401
        except ImportError:
            raise ImportError("Run: pip install TTS")

        _patch_torch_isin()
        _patch_xtts_stream_modules()

        manager = ModelManager()
        model_path, config_path = _resolve_model_paths(
            manager,
            "tts_models/multilingual/multi-dataset/xtts_v2",
        )

        config = XttsConfig()
        config.load_json(config_path)

        self._model = Xtts.init_from_config(config)
        self._model.load_checkpoint(config, checkpoint_dir=model_path, eval=True)

        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model.to(self._device)
        log.info("TTS fallback model | provider=xtts | model=xtts_v2 | device=%s", self._device)

        self._gpt_cond_latent = None
        self._speaker_embedding = None
        self._reference = self._validate_reference()
        self._cache_speaker_embedding()

    def _should_stream(self) -> bool:
        if XTTS_STREAMING_MODE == "on":
            return True
        if XTTS_STREAMING_MODE == "off":
            return False
        return self._device == "cuda"

    def _validate_reference(self) -> Optional[str]:
        path = Path(VOICE_SAMPLE)
        if not path.exists():
            log.warning(
                "Voice sample not found at '%s'. Falling back to XTTS default voice.",
                VOICE_SAMPLE,
            )
            return None
        return str(path)

    def _cache_speaker_embedding(self) -> None:
        if not self._reference:
            return
        with torch.inference_mode():
            gpt_cond_latent, speaker_embedding = self._model.get_conditioning_latents(
                audio_path=[self._reference]
            )
        self._gpt_cond_latent = gpt_cond_latent
        self._speaker_embedding = speaker_embedding

    def _xtts_lang(self, lang: str) -> str:
        lang_map = {
            "zh": "zh-cn",
            "en": "en",
            "ja": "ja",
            "ko": "ko",
            "fr": "fr",
            "de": "de",
            "es": "es",
        }
        return lang_map.get(lang, "en")

    def synthesize_streaming(self, text: str, lang: str) -> bool:
        if not self._should_stream():
            return False
        try:
            self._stream_to_speaker(text, lang)
            return True
        except Exception as exc:
            log.warning("Streaming XTTS failed, falling back to file mode: %s", exc)
            return False

    def _stream_to_speaker(self, text: str, lang: str) -> None:
        if self._gpt_cond_latent is None or self._speaker_embedding is None:
            raise RuntimeError("No cached speaker embedding available")

        import sounddevice as sd

        xtts_lang = self._xtts_lang(lang)
        start_time = time.perf_counter()
        chunk_count = 0
        audio_q: queue.Queue[Optional[np.ndarray]] = queue.Queue(maxsize=8)
        playback_started = threading.Event()
        buffered_chunks = 0

        def _player() -> None:
            with sd.OutputStream(
                samplerate=self.SAMPLE_RATE,
                channels=1,
                dtype="int16",
            ) as stream:
                playback_started.wait()
                while True:
                    chunk = audio_q.get()
                    if chunk is None:
                        break
                    stream.write(chunk)

        player_thread = threading.Thread(target=_player, daemon=True, name="xtts-player")
        player_thread.start()

        try:
            with torch.inference_mode():
                chunks = self._model.inference_stream(
                    text,
                    xtts_lang,
                    self._gpt_cond_latent,
                    self._speaker_embedding,
                    temperature=XTTS_TEMPERATURE,
                    length_penalty=XTTS_LENGTH_PENALTY,
                    repetition_penalty=XTTS_REPETITION_PENALTY,
                    top_k=XTTS_TOP_K,
                    top_p=XTTS_TOP_P,
                    speed=XTTS_SPEED,
                    stream_chunk_size=XTTS_STREAM_CHUNK_SIZE,
                    enable_text_splitting=False,
                )
                for index, chunk in enumerate(chunks):
                    buffered_chunks += 1
                    audio_np = (chunk.squeeze().cpu().numpy() * 32767).astype(np.int16)
                    audio_q.put(audio_np)
                    if (
                        not playback_started.is_set()
                        and buffered_chunks >= XTTS_STREAM_PREROLL_CHUNKS
                    ):
                        playback_started.set()
                    chunk_count += 1
                if not playback_started.is_set():
                    playback_started.set()
        finally:
            audio_q.put(None)
            player_thread.join()

        _ = (chunk_count, start_time)

    def synthesize_to_file(self, text: str, lang: str, output_path: str) -> bool:
        xtts_lang = self._xtts_lang(lang)
        try:
            if self._gpt_cond_latent is not None and self._speaker_embedding is not None:
                with torch.inference_mode():
                    out = self._model.inference(
                        text,
                        xtts_lang,
                        self._gpt_cond_latent,
                        self._speaker_embedding,
                        temperature=XTTS_TEMPERATURE,
                        length_penalty=XTTS_LENGTH_PENALTY,
                        repetition_penalty=XTTS_REPETITION_PENALTY,
                        top_k=XTTS_TOP_K,
                        top_p=XTTS_TOP_P,
                        speed=XTTS_SPEED,
                        enable_text_splitting=False,
                    )
            else:
                with torch.inference_mode():
                    out = self._model.inference(
                        text,
                        xtts_lang,
                        temperature=XTTS_TEMPERATURE,
                    )

            audio = np.array(out["wav"], dtype=np.float32)
            audio = np.clip(audio, -1.0, 1.0)
            audio_int16 = (audio * 32767).astype(np.int16)

            import wave

            with wave.open(output_path, "wb") as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)
                wav_file.setframerate(self.SAMPLE_RATE)
                wav_file.writeframes(audio_int16.tobytes())
            return True
        except Exception as exc:
            log.error("XTTS file synthesis failed: %s", exc)
            return False


# =============================================================================
# Backend: OpenVoice v2
# =============================================================================

class _OpenVoiceBackend:
    def __init__(self):
        try:
            from openvoice import se_extractor
            from openvoice.api import ToneColorConverter
            from melo.api import TTS as MeloTTS
        except ImportError:
            raise ImportError("Run: pip install git+https://github.com/myshell-ai/OpenVoice")

        self._se_extractor = se_extractor
        self._ToneColorConverter = ToneColorConverter
        self._MeloTTS = MeloTTS
        self._device = "cuda" if torch.cuda.is_available() else "cpu"

        ckpt_dir = "checkpoints_v2/converter"
        if not Path(ckpt_dir).exists():
            raise FileNotFoundError(
                f"Checkpoints not found at '{ckpt_dir}'.\n"
                "Download from https://huggingface.co/myshell-ai/OpenVoiceV2"
            )

        self._converter = ToneColorConverter(ckpt_dir, device=self._device)
        self._converter.load_ckpt(os.path.join(ckpt_dir, "checkpoint.pth"))
        self._reference_se = self._load_reference_se()
        log.info("TTS fallback model | provider=openvoice | device=%s", self._device)

    def _load_reference_se(self):
        path = Path(VOICE_SAMPLE)
        if not path.exists():
            log.warning("Voice sample not found. Tone cloning disabled.")
            return None
        se, _ = self._se_extractor.get_se(str(path), self._converter.model, vad=True)
        return se

    def synthesize(self, text: str, lang: str, output_path: str) -> bool:
        melo_lang_map = {
            "en": "EN",
            "zh": "ZH",
            "ja": "JP",
            "ko": "KR",
            "fr": "FR",
            "de": "DE",
        }
        melo_lang = melo_lang_map.get(lang, "EN")
        try:
            base_tts = self._MeloTTS(language=melo_lang, device=self._device)
            speaker_id = next(iter(base_tts.hps.data.spk2id.values()))
            tmp_base = output_path + ".base.wav"
            base_tts.tts_to_file(text, speaker_id, tmp_base, speed=1.0)

            if self._reference_se is not None:
                target_se, _ = self._se_extractor.get_se(
                    tmp_base,
                    self._converter.model,
                    vad=False,
                )
                self._converter.convert(
                    audio_src_path=tmp_base,
                    src_se=target_se,
                    tgt_se=self._reference_se,
                    output_path=output_path,
                    tau=0.3,
                )
                if Path(tmp_base).exists():
                    os.remove(tmp_base)
            else:
                import shutil

                shutil.move(tmp_base, output_path)

            return True
        except Exception as exc:
            log.error("OpenVoice synthesis failed: %s", exc)
            return False


# =============================================================================
# Backend: edge-tts
# =============================================================================

class _EdgeTTSBackend:
    def synthesize_sync(self, text: str, lang: str, output_path: str) -> bool:
        future = asyncio.run_coroutine_threadsafe(
            self._generate(text, lang, output_path),
            _loop,
        )
        try:
            return future.result(timeout=20)
        except Exception as exc:
            log.error("edge-tts error: %s", exc)
            return False

    async def _generate(self, text: str, lang: str, output_path: str) -> bool:
        import edge_tts as _edge

        voice = EDGE_VOICE_MAP.get(lang, "en-US-AriaNeural")
        for attempt in range(1, RETRY_COUNT + 1):
            try:
                comm = _edge.Communicate(text, voice, proxy=PROXY)
                await comm.save(output_path)
                if Path(output_path).stat().st_size < 100:
                    raise ValueError("File too small")
                return True
            except Exception as exc:
                log.warning("edge-tts attempt %d/%d: %s", attempt, RETRY_COUNT, exc)
                if attempt < RETRY_COUNT:
                    await asyncio.sleep(RETRY_DELAY)
        return False


class _QwenTTSAPIBackend:
    def __init__(self):
        if not DASHSCOPE_API_KEY:
            raise ValueError(
                "USE_QWEN_TTS_API is enabled but DASHSCOPE_API_KEY is not configured."
            )
        if not QWEN_TTS_VOICE:
            raise ValueError(
                "USE_QWEN_TTS_API is enabled but QWEN_TTS_VOICE is not configured."
            )
        try:
            import dashscope
            import sounddevice as sd
            from dashscope.audio.qwen_tts_realtime import (
                AudioFormat,
                QwenTtsRealtime,
                QwenTtsRealtimeCallback,
            )
        except ImportError as exc:
            raise ImportError("dashscope with qwen_tts_realtime support is required.") from exc

        class _RealtimeCallback(QwenTtsRealtimeCallback):
            def __init__(self, owner):
                super().__init__()
                self._owner = owner

            def on_open(self) -> None:
                self._owner._handle_open()

            def on_close(self, close_status_code, close_msg) -> None:
                self._owner._handle_close(close_status_code, close_msg)

            def on_event(self, response: str) -> None:
                self._owner._handle_event(response)

        self._dashscope = dashscope
        self._dashscope.api_key = DASHSCOPE_API_KEY
        self._sd = sd
        self._AudioFormat = AudioFormat
        self._QwenTtsRealtime = QwenTtsRealtime
        self._model = QWEN_TTS_MODEL
        self._voice = QWEN_TTS_VOICE
        self._url = QWEN_TTS_URL
        self._session_mode = (
            QWEN_TTS_SESSION_MODE
            if QWEN_TTS_SESSION_MODE in {"commit", "server_commit"}
            else "commit"
        )
        self._timeout_sec = max(5.0, QWEN_TTS_RESPONSE_TIMEOUT_SEC)
        self._callback = _RealtimeCallback(self)
        self._tts = None
        self._session_created = threading.Event()
        self._session_updated = threading.Event()
        self._response_done = threading.Event()
        self._playback_finished = threading.Event()
        self._player_stop = threading.Event()
        self._state_lock = threading.RLock()
        self._player_thread = None
        self._audio_queue = None
        self._response_error = None
        self._first_audio_at = None
        self._response_started_at = None
        log.info(
            "TTS primary model | provider=qwen_api_realtime | model=%s | voice_configured=%s | mode=%s",
            self._model,
            bool(self._voice),
            self._session_mode,
        )

    def synthesize_and_play(self, text: str, lang: str, voice: str | None = None) -> TTSTiming:
        desired_voice = (voice or self._voice or "").strip()
        if not desired_voice:
            raise ValueError("No Qwen TTS voice is configured for realtime synthesis.")
        with self._state_lock:
            self._reset_response_state()
            self._connect(desired_voice)
            self._start_player()
            self._response_started_at = time.perf_counter()
            try:
                self._tts.clear_appended_text()
            except Exception:
                pass
            self._tts.append_text(text)
            self._tts.commit()

        try:
            if not self._response_done.wait(timeout=self._timeout_sec):
                self._stop_player()
                raise TimeoutError(
                    f"Qwen realtime TTS timed out after {self._timeout_sec:.1f}s waiting for response.done."
                )
            if self._response_error is not None:
                self._stop_player()
                raise RuntimeError(self._response_error)
            if not self._playback_finished.wait(timeout=self._timeout_sec):
                self._stop_player()
                raise TimeoutError(
                    f"Qwen realtime TTS playback did not finish within {self._timeout_sec:.1f}s."
                )

            provider_ready_ms = 0.0
            if self._first_audio_at is not None and self._response_started_at is not None:
                provider_ready_ms = (self._first_audio_at - self._response_started_at) * 1000.0
            total_ms = 0.0
            if self._response_started_at is not None:
                total_ms = (time.perf_counter() - self._response_started_at) * 1000.0
            return TTSTiming(
                provider="qwen_api",
                mode="streaming",
                provider_ready_ms=provider_ready_ms,
                total_ms=total_ms,
            )
        finally:
            with self._state_lock:
                self._safe_close()

    def _connect(self, voice: str) -> None:
        self._session_created.clear()
        self._session_updated.clear()
        self._tts = self._QwenTtsRealtime(
            model=self._model,
            callback=self._callback,
            url=self._url,
        )
        self._tts.connect()
        if not self._session_created.wait(timeout=5.0):
            raise TimeoutError("Qwen realtime TTS session was not created within 5s.")
        self._tts.update_session(
            voice=voice,
            response_format=self._AudioFormat.PCM_24000HZ_MONO_16BIT,
            mode=self._session_mode,
        )
        if not self._session_updated.wait(timeout=5.0):
            raise TimeoutError("Qwen realtime TTS session was not updated within 5s.")

    def _reset_response_state(self) -> None:
        self._response_done.clear()
        self._playback_finished.clear()
        self._player_stop.clear()
        self._audio_queue = queue.Queue()
        self._player_thread = None
        self._response_error = None
        self._first_audio_at = None
        self._response_started_at = None

    def _start_player(self) -> None:
        self._player_thread = threading.Thread(
            target=self._play_audio_stream,
            daemon=True,
            name="qwen-tts-realtime-player",
        )
        self._player_thread.start()

    def _stop_player(self) -> None:
        self._player_stop.set()
        if self._audio_queue is not None:
            self._audio_queue.put(None)
        if self._player_thread is not None:
            self._player_thread.join(timeout=2.0)
        self._playback_finished.set()

    def _play_audio_stream(self) -> None:
        playback_bus.begin_playback()
        try:
            with self._sd.RawOutputStream(
                samplerate=QWEN_TTS_SAMPLE_RATE,
                channels=1,
                dtype="int16",
                blocksize=QWEN_TTS_STREAM_BLOCKSIZE,
            ) as stream:
                while not self._player_stop.is_set():
                    chunk = self._audio_queue.get()
                    if chunk is None:
                        break
                    if not chunk:
                        continue
                    stream.write(chunk)
                    samples = np.frombuffer(chunk, dtype=np.int16).astype(np.float32) / 32768.0
                    playback_bus.push_render_frame(samples, sample_rate=QWEN_TTS_SAMPLE_RATE)
                stream.stop()
        finally:
            playback_bus.end_playback()
            self._playback_finished.set()

    def _handle_open(self) -> None:
        pass

    def _handle_close(self, close_status_code, close_msg) -> None:
        if not self._response_done.is_set():
            self._response_error = (
                f"Qwen realtime TTS connection closed unexpectedly: code={close_status_code}, msg={close_msg}"
            )
            self._response_done.set()
        if self._audio_queue is not None:
            self._audio_queue.put(None)

    def _handle_event(self, response) -> None:
        try:
            event_type = response.get("type")
            if event_type == "session.created":
                self._session_created.set()
                return
            if event_type == "session.updated":
                self._session_updated.set()
                return
            if event_type == "response.audio.delta":
                delta = response.get("delta")
                if not delta:
                    return
                if self._first_audio_at is None:
                    self._first_audio_at = time.perf_counter()
                self._audio_queue.put(base64.b64decode(delta))
                return
            if event_type == "response.done":
                self._response_done.set()
                if self._audio_queue is not None:
                    self._audio_queue.put(None)
                return
            if event_type == "error":
                self._response_error = json.dumps(response, ensure_ascii=False)
                self._response_done.set()
                if self._audio_queue is not None:
                    self._audio_queue.put(None)
        except Exception as exc:
            self._response_error = f"Qwen realtime TTS callback error: {exc}"
            self._response_done.set()
            if self._audio_queue is not None:
                self._audio_queue.put(None)

    def _safe_close(self) -> None:
        try:
            if self._tts is not None:
                self._tts.close()
        except Exception:
            pass
        self._tts = None


# =============================================================================
# Backend loader
# =============================================================================

def _load_backend():
    if TTS_BACKEND == "xtts":
        return _XTTSBackend()
    if TTS_BACKEND == "openvoice":
        return _OpenVoiceBackend()
    if TTS_BACKEND == "edge":
        return _EdgeTTSBackend()
    raise ValueError(f"Unknown TTS_BACKEND '{TTS_BACKEND}'.")


_qwen_tts_backend = None
_backend = None

if USE_QWEN_TTS_API:
    log.info("startup | loading primary backend=qwen_api")
    try:
        _qwen_tts_backend = _QwenTTSAPIBackend()
    except Exception as exc:
        if API_ONLY_TTS:
            raise RuntimeError(
                f"Qwen TTS API backend failed to initialize in API-only mode: {exc}"
            ) from exc
        log.warning(
            "Qwen TTS API backend disabled at startup, local backend will be used instead: %s",
            exc,
        )
elif API_ONLY_TTS:
    raise RuntimeError("TTS_MODE=api_only requires USE_QWEN_TTS_API=true")

if not API_ONLY_TTS:
    log.info("startup | loading fallback backend=%s", TTS_BACKEND)
    _backend = _load_backend()

log.info(
    "TTS startup | mode=%s | primary=%s | fallback=%s",
    TTS_MODE,
    "qwen_api" if _qwen_tts_backend is not None else "none",
    TTS_BACKEND if _backend is not None else "none",
)


# =============================================================================
# Playback helper for file-based backends
# =============================================================================

def _play_file(path: str) -> None:
    play_audio_file(path, logger=log)


# =============================================================================
# Public API
# =============================================================================

def speak(text: str, lang: str = "en", voice: str | None = None) -> TTSTiming | None:
    if not text or not text.strip():
        return None

    start_time = time.perf_counter()
    try:
        if USE_QWEN_TTS_API and _qwen_tts_backend is not None:
            try:
                timing = _qwen_tts_backend.synthesize_and_play(text, lang, voice=voice)
                log.info(
                    "TTS provider | provider=qwen_api | model=%s | mode=%s | voice=%s",
                    QWEN_TTS_MODEL,
                    timing.mode,
                    voice or QWEN_TTS_VOICE or "default",
                )
                return timing
            except Exception as exc:
                if API_ONLY_TTS:
                    log.error("Qwen TTS API failed and API-only mode is enabled: %s", exc)
                    return None
                log.warning("Qwen TTS API failed, falling back to local backend: %s", exc)

        if _backend is None:
            log.error("No TTS backend available for fallback.")
            return None

        if TTS_BACKEND == "xtts":
            if _backend.synthesize_streaming(text, lang):
                log.info("TTS provider | provider=xtts | mode=streaming")
                total_ms = (time.perf_counter() - start_time) * 1000.0
                return TTSTiming(
                    provider="xtts",
                    mode="streaming",
                    provider_ready_ms=total_ms,
                    total_ms=total_ms,
                )

            tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            path = tmp.name
            tmp.close()
            try:
                ok = _backend.synthesize_to_file(text, lang, path)
                if not ok:
                    log.error("XTTS fallback synthesis failed - skipping.")
                    return None
                size = Path(path).stat().st_size if Path(path).exists() else 0
                if size < 100:
                    log.error("XTTS fallback output too small (%d B).", size)
                    return None
                provider_ready_ms = (time.perf_counter() - start_time) * 1000.0
                log.info("TTS provider | provider=xtts | mode=file")
                _play_file(path)
                total_ms = (time.perf_counter() - start_time) * 1000.0
                return TTSTiming(
                    provider="xtts",
                    mode="file",
                    provider_ready_ms=provider_ready_ms,
                    total_ms=total_ms,
                )
            finally:
                try:
                    if os.path.exists(path):
                        os.remove(path)
                except OSError:
                    pass

        suffix = ".wav" if TTS_BACKEND == "openvoice" else ".mp3"
        tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
        path = tmp.name
        tmp.close()

        try:
            if TTS_BACKEND == "edge":
                ok = _backend.synthesize_sync(text, lang, path)
            else:
                ok = _backend.synthesize(text, lang, path)

            if not ok:
                log.error("Synthesis failed - skipping.")
                return None

            size = Path(path).stat().st_size if Path(path).exists() else 0
            if size < 100:
                log.error("Output too small (%d B).", size)
                return None

            provider_ready_ms = (time.perf_counter() - start_time) * 1000.0
            log.info("TTS provider | provider=%s", TTS_BACKEND)
            _play_file(path)
            total_ms = (time.perf_counter() - start_time) * 1000.0
            return TTSTiming(
                provider=TTS_BACKEND,
                mode="file",
                provider_ready_ms=provider_ready_ms,
                total_ms=total_ms,
            )
        finally:
            try:
                if os.path.exists(path):
                    os.remove(path)
            except OSError:
                pass

    except Exception as exc:
        log.error("speak() error: %s", exc)
        return None


# =============================================================================
# Optional: standalone Redis-listener service
# =============================================================================

REDIS_HOST = "localhost"
REDIS_PORT = 6379
INPUT_CHANNEL = "channel:mt_to_tts"
MAX_QUEUE = 3


def run_as_service() -> None:
    redis_client = redis.Redis(
        host=REDIS_HOST,
        port=REDIS_PORT,
        decode_responses=True,
    )
    pubsub = redis_client.pubsub()
    pubsub.subscribe(INPUT_CHANNEL)
    log.info(
        "TTS service ready | mode=%s | primary=%s | fallback=%s",
        TTS_MODE,
        "qwen_api" if _qwen_tts_backend is not None else "none",
        TTS_BACKEND if _backend is not None else "none",
    )

    pending: list[tuple[str, str]] = []
    try:
        while True:
            msg = pubsub.get_message(timeout=0.1)
            if msg and msg["type"] == "message":
                raw = msg["data"]
                try:
                    data = json.loads(raw)
                    text = data.get("text", "").strip()
                    lang = data.get("lang", "en")
                except (json.JSONDecodeError, TypeError):
                    text, lang = str(raw).strip(), "en"

                if text:
                    pending.append((text, lang))
                    if len(pending) > MAX_QUEUE:
                        pending.pop(0)

            if pending:
                speak(*pending.pop(0))
    except KeyboardInterrupt:
        log.info("Shutting down.")
    finally:
        pubsub.unsubscribe()
        _loop.call_soon_threadsafe(_loop.stop)


if __name__ == "__main__":
    run_as_service()
