"""
api.py - dashboard server + optional local translation API + orchestrator control
"""

import logging
import os
import subprocess
import sys
import threading
from collections import deque
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv, set_key
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, field_validator

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
WEB_DIR = BASE_DIR / "web"
ORCHESTRATOR_PATH = BASE_DIR / "orchestrator.py"
ENV_PATH = BASE_DIR / ".env"
MAX_PIPELINE_LOG_LINES = 300

sys.path.append(str(BASE_DIR))
from translator import LightTranslator


logging.basicConfig(level=logging.INFO, format="%(asctime)s [API] %(message)s")
log = logging.getLogger(__name__)

SUPPORTED_LANGS = {
    "en",
    "zh",
    "ja",
    "ko",
    "fr",
    "de",
    "es",
    "ar",
    "ru",
    "pt",
    "it",
}

LANGUAGE_NAME_MAP = {
    "zh": "Chinese",
    "en": "English",
    "ja": "Japanese",
    "ko": "Korean",
    "fr": "French",
    "de": "German",
    "es": "Spanish",
    "ar": "Arabic",
    "ru": "Russian",
    "pt": "Portuguese",
    "it": "Italian",
}

_translator: LightTranslator | None = None
_translator_lock = threading.Lock()

_pipeline_process: subprocess.Popen | None = None
_pipeline_lock = threading.Lock()
_pipeline_logs: deque[str] = deque(maxlen=MAX_PIPELINE_LOG_LINES)
_pipeline_reader_thread: threading.Thread | None = None


def _append_pipeline_log(line: str) -> None:
    _pipeline_logs.append(line.rstrip())


def _ensure_translator() -> LightTranslator:
    global _translator
    if _translator is not None:
        return _translator

    with _translator_lock:
        if _translator is None:
            log.info("Loading local translation model on demand...")
            _translator = LightTranslator()
            log.info("Local translation model ready.")

    return _translator


def _pipeline_status_value() -> str:
    if _pipeline_process is None:
        return "stopped"
    if _pipeline_process.poll() is None:
        return "running"
    return "stopped"


def _read_pipeline_output(proc: subprocess.Popen) -> None:
    global _pipeline_process
    try:
        if proc.stdout is None:
            return
        for line in proc.stdout:
            _append_pipeline_log(line)
    finally:
        with _pipeline_lock:
            if _pipeline_process is proc and proc.poll() is not None:
                _append_pipeline_log(f"[dashboard] orchestrator exited with code {proc.returncode}")
                _pipeline_process = None


def _start_pipeline_process() -> dict:
    global _pipeline_process, _pipeline_reader_thread

    with _pipeline_lock:
        if _pipeline_process is not None and _pipeline_process.poll() is None:
            return {
                "status": "running",
                "message": "orchestrator is already running",
                "pid": _pipeline_process.pid,
            }

        _pipeline_logs.clear()
        _append_pipeline_log("[dashboard] starting orchestrator...")

        child_env = os.environ.copy()
        child_env["PYTHONUNBUFFERED"] = "1"

        proc = subprocess.Popen(
            [sys.executable, "-u", str(ORCHESTRATOR_PATH)],
            cwd=str(BASE_DIR),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            env=child_env,
        )
        _pipeline_process = proc
        _append_pipeline_log(f"[dashboard] orchestrator pid={proc.pid}")
        _pipeline_reader_thread = threading.Thread(
            target=_read_pipeline_output,
            args=(proc,),
            daemon=True,
            name="orchestrator-log-reader",
        )
        _pipeline_reader_thread.start()

        return {
            "status": "running",
            "message": "orchestrator started",
            "pid": proc.pid,
        }


def _stop_pipeline_process() -> dict:
    global _pipeline_process

    with _pipeline_lock:
        proc = _pipeline_process
        if proc is None or proc.poll() is not None:
            _pipeline_process = None
            return {
                "status": "stopped",
                "message": "orchestrator is not running",
                "pid": None,
            }

        _append_pipeline_log("[dashboard] stopping orchestrator...")
        proc.terminate()
        pid = proc.pid

    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        _append_pipeline_log("[dashboard] terminate timed out, killing orchestrator...")
        proc.kill()
        proc.wait(timeout=5)

    with _pipeline_lock:
        if _pipeline_process is proc:
            _pipeline_process = None

    _append_pipeline_log(f"[dashboard] orchestrator stopped (pid={pid})")
    return {
        "status": "stopped",
        "message": "orchestrator stopped",
        "pid": pid,
    }


def _build_pipeline_payload() -> dict:
    pid = None
    if _pipeline_process is not None and _pipeline_process.poll() is None:
        pid = _pipeline_process.pid

    return {
        "status": _pipeline_status_value(),
        "pid": pid,
        "logs": list(_pipeline_logs),
    }


@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("API server ready.")
    yield
    log.info("Shutting down.")


app = FastAPI(title="Realtime Speech Dashboard", lifespan=lifespan)

if WEB_DIR.exists():
    app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")


class TranslationRequest(BaseModel):
    text: str
    source_lang: str = "en"
    target_lang: str = "zh"

    @field_validator("source_lang", "target_lang")
    @classmethod
    def validate_lang(cls, value: str) -> str:
        value = value.lower().strip()
        if value not in SUPPORTED_LANGS:
            raise ValueError(
                f"Unsupported language '{value}'. Supported: {sorted(SUPPORTED_LANGS)}"
            )
        return value

    @field_validator("text")
    @classmethod
    def validate_text(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("text must not be empty")
        if len(value) > 2000:
            raise ValueError("text exceeds 2000-character limit")
        return value


class LanguageConfigUpdate(BaseModel):
    source_lang: str
    target_lang: str

    @field_validator("source_lang", "target_lang")
    @classmethod
    def validate_lang(cls, value: str) -> str:
        value = value.lower().strip()
        if value not in SUPPORTED_LANGS:
            raise ValueError(
                f"Unsupported language '{value}'. Supported: {sorted(SUPPORTED_LANGS)}"
            )
        return value


def _build_stack_payload() -> dict:
    use_qwen_asr = os.getenv("USE_QWEN_ASR_API", "false").lower() == "true"
    use_local_mt = os.getenv("USE_LOCAL_MT", "false").lower() == "true"
    use_qwen_tts = os.getenv("USE_QWEN_TTS_API", "false").lower() == "true"

    mt_source = os.getenv("MT_SOURCE_LANG", "zh")
    mt_target = os.getenv("MT_TARGET_LANG", "en")

    return {
        "routes": {
            "asr_primary": "qwen_api" if use_qwen_asr else "zipformer",
            "asr_fallback": "zipformer" if use_qwen_asr else None,
            "mt_primary": "local_api" if use_local_mt else "deepseek",
            "tts_primary": "qwen_api" if use_qwen_tts else os.getenv("TTS_BACKEND", "xtts"),
            "tts_fallback": os.getenv("TTS_BACKEND", "xtts") if use_qwen_tts else None,
        },
        "models": {
            "asr": os.getenv("QWEN_ASR_MODEL", "qwen3-asr-flash-realtime-2025-10-27")
            if use_qwen_asr
            else "sherpa-onnx zipformer",
            "mt": os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
            if not use_local_mt
            else "local translation API",
            "tts": os.getenv("QWEN_TTS_MODEL", "qwen3-tts-vc-2026-01-22")
            if use_qwen_tts
            else os.getenv("TTS_BACKEND", "xtts"),
            "tts_fallback": os.getenv("TTS_BACKEND", "xtts") if use_qwen_tts else None,
        },
        "languages": {
            "asr": {
                "code": os.getenv("QWEN_ASR_LANGUAGE", "zh"),
                "name": LANGUAGE_NAME_MAP.get(
                    os.getenv("QWEN_ASR_LANGUAGE", "zh"),
                    os.getenv("QWEN_ASR_LANGUAGE", "zh"),
                ),
            },
            "translation": {
                "source_code": mt_source,
                "source_name": LANGUAGE_NAME_MAP.get(mt_source, mt_source),
                "target_code": mt_target,
                "target_name": LANGUAGE_NAME_MAP.get(mt_target, mt_target),
            },
        },
        "voice": {
            "configured": bool(os.getenv("QWEN_TTS_VOICE", "").strip()),
            "reference_sample": os.getenv("VOICE_SAMPLE", "voice_samples/my_voice.wav"),
        },
    }


@app.get("/")
async def index():
    index_path = WEB_DIR / "index.html"
    if not index_path.exists():
        raise HTTPException(status_code=404, detail="Frontend not found.")
    return FileResponse(index_path)


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "local_translation_model_loaded": _translator is not None,
    }


@app.get("/api/stack")
async def stack():
    return _build_stack_payload()


@app.post("/api/config/languages")
async def update_language_config(payload: LanguageConfigUpdate):
    if not ENV_PATH.exists():
        raise HTTPException(status_code=500, detail=".env file not found.")

    try:
        set_key(str(ENV_PATH), "QWEN_ASR_LANGUAGE", payload.source_lang)
        set_key(str(ENV_PATH), "MT_SOURCE_LANG", payload.source_lang)
        set_key(str(ENV_PATH), "MT_TARGET_LANG", payload.target_lang)
        os.environ["QWEN_ASR_LANGUAGE"] = payload.source_lang
        os.environ["MT_SOURCE_LANG"] = payload.source_lang
        os.environ["MT_TARGET_LANG"] = payload.target_lang
        return {
            "status": "success",
            "qwen_asr_language": payload.source_lang,
            "source_lang": payload.source_lang,
            "target_lang": payload.target_lang,
            "message": "Updated QWEN_ASR_LANGUAGE and MT_SOURCE_LANG together.",
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/api/pipeline/status")
async def pipeline_status():
    return _build_pipeline_payload()


@app.post("/api/pipeline/start")
async def pipeline_start():
    return _start_pipeline_process()


@app.post("/api/pipeline/stop")
async def pipeline_stop():
    return _stop_pipeline_process()


@app.post("/translate")
async def translate_text(req: TranslationRequest):
    try:
        translator = _ensure_translator()
        result = translator.translate(
            req.text,
            source_lang=req.source_lang,
            target_lang=req.target_lang,
        )
        return {
            "status": "success",
            "original_text": req.text,
            "translated_text": result,
            "source_lang": req.source_lang,
            "target_lang": req.target_lang,
        }
    except Exception as exc:
        log.exception("Translation failed")
        raise HTTPException(status_code=500, detail=str(exc))
