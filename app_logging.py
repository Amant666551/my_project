from __future__ import annotations

import logging
import os
import sys
import threading
import time
from datetime import datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path


_CONFIGURED = False
_HANDLER_TAG = "_pipeline_logging_handler"
_LOGGER_TAG = "_pipeline_named_logger"
_APP_LOGGER_PREFIX = "pipeline"
_CONSOLE_HANDLER: logging.Handler | None = None
_FILE_HANDLER: logging.Handler | None = None


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


def _log_level(name: str, default: int = logging.INFO) -> int:
    value = os.getenv(name, "").strip().upper()
    if not value:
        return default
    return getattr(logging, value, default)


def _daily_log_only() -> bool:
    return _env_bool("LOG_DAILY_ONLY", True)


def _log_rotate_minutes() -> int:
    return max(0, _env_int("LOG_ROTATE_MINUTES", 0))


def _console_mode() -> str:
    value = os.getenv("LOG_CONSOLE_MODE", "minimal").strip().lower()
    return _normalize_mode(value, default="minimal")


def _file_mode() -> str:
    value = os.getenv("LOG_FILE_MODE", "minimal").strip().lower()
    return _normalize_mode(value, default="minimal")


def _normalize_mode(value: str, *, default: str) -> str:
    aliases = {
        "quiet": "minimal",
        "concise": "minimal",
        "standard": "normal",
        "default": "normal",
        "full": "verbose",
    }
    normalized = aliases.get(value, value)
    if normalized not in {"minimal", "normal", "verbose"}:
        return default
    return normalized


def _is_core_result_record(record: logging.LogRecord, message: str) -> bool:
    if record.name == f"{_APP_LOGGER_PREFIX}.ASR" and message.startswith("final | text="):
        return True
    if record.name == f"{_APP_LOGGER_PREFIX}.MT" and message.startswith("result | text="):
        return True
    if record.name == f"{_APP_LOGGER_PREFIX}.TURN" and message.startswith("turn | "):
        return True
    if record.name == f"{_APP_LOGGER_PREFIX}.TTS" and message.startswith("TTS provider |"):
        return True
    if record.name == f"{_APP_LOGGER_PREFIX}.PIPELINE" and message == "System ready - start speaking!":
        return True
    return False


def _is_pipeline_lifecycle_record(record: logging.LogRecord, message: str) -> bool:
    return record.name == f"{_APP_LOGGER_PREFIX}.PIPELINE" and message in {
        "ASR -> MT -> TTS pipeline starting",
        "System ready - start speaking!",
        "Pipeline stopped.",
    }


def _is_mt_context_record(record: logging.LogRecord, message: str) -> bool:
    return record.name == f"{_APP_LOGGER_PREFIX}.MT.CONTEXT" and (
        message.startswith("deepseek_call |")
        or message.startswith("deepseek_done |")
        or message.startswith("scene_result |")
        or message.startswith("scene_analyzer |")
    )


def _is_latency_record(record: logging.LogRecord, message: str) -> bool:
    return record.name == f"{_APP_LOGGER_PREFIX}.LATENCY" and message.startswith("trace |")


def _is_asr_partial_record(record: logging.LogRecord, message: str) -> bool:
    return record.name == f"{_APP_LOGGER_PREFIX}.ASR" and message.startswith("partial | text=")


class _MinimalConsoleFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        message = record.getMessage()
        return _is_core_result_record(record, message)


class _NormalConsoleFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        if record.levelno >= logging.WARNING:
            return True
        message = record.getMessage()
        return _is_core_result_record(record, message) or _is_pipeline_lifecycle_record(record, message)


class _MinimalFileFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        if record.levelno >= logging.WARNING:
            return True

        message = record.getMessage()
        if _is_core_result_record(record, message):
            return True
        if _is_latency_record(record, message):
            return True
        return False


class _NormalFileFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        if record.levelno >= logging.WARNING:
            return True

        message = record.getMessage()
        if _is_core_result_record(record, message):
            return True
        if _is_pipeline_lifecycle_record(record, message):
            return True
        if _is_mt_context_record(record, message):
            return True
        if _is_latency_record(record, message):
            return True
        if _is_asr_partial_record(record, message):
            return True

        return False


class _MinimalConsoleFormatter(logging.Formatter):
    @staticmethod
    def _clean_text(text: str) -> str:
        return text.replace("\r", " ").replace("\n", " ").strip()

    def format(self, record: logging.LogRecord) -> str:
        message = self._clean_text(record.getMessage())
        if record.name == f"{_APP_LOGGER_PREFIX}.PIPELINE" and message == "System ready - start speaking!":
            return "请输入："
        if record.name == f"{_APP_LOGGER_PREFIX}.ASR" and message.startswith("final | text="):
            return f"[ASR final  ]: {message.removeprefix('final | text=')}"
        if record.name == f"{_APP_LOGGER_PREFIX}.MT" and message.startswith("result | text="):
            return f"[MT  ]: {message.removeprefix('result | text=')}"
        if record.name == f"{_APP_LOGGER_PREFIX}.TURN" and message.startswith("turn | "):
            return f"{self.formatTime(record, self.datefmt)} [TURN] {message}"
        if record.name == f"{_APP_LOGGER_PREFIX}.TTS" and message.startswith("TTS provider |"):
            return f"{self.formatTime(record, self.datefmt)} [TTS] {message}"
        return message


def default_log_path() -> Path:
    base_dir = Path(__file__).resolve().parent
    log_dir = Path(os.getenv("LOG_DIR", str(base_dir / "logs")))
    file_name = os.getenv("LOG_FILE_NAME", "pipeline.log").strip() or "pipeline.log"
    if _log_rotate_minutes() > 0:
        return log_dir / file_name
    if not _daily_log_only():
        return log_dir / file_name

    base_name = Path(file_name)
    today = datetime.now().strftime("%Y-%m-%d")
    daily_name = f"{base_name.stem}-{today}{base_name.suffix or '.log'}"
    return log_dir / daily_name


def _cleanup_expired_daily_logs(log_dir: Path, active_log_path: Path) -> None:
    if not _daily_log_only():
        return

    try:
        active_name = active_log_path.name
        active_prefix = active_name.split("-", 1)[0]
        for candidate in log_dir.iterdir():
            if not candidate.is_file():
                continue
            if candidate == active_log_path:
                continue
            if not candidate.name.startswith(active_prefix):
                continue
            try:
                candidate.unlink()
            except OSError:
                continue
    except OSError:
        return


class _PeriodicTruncateFileHandler(logging.FileHandler):
    def __init__(self, filename: Path, interval_minutes: int, encoding: str = "utf-8"):
        self.interval_seconds = max(60, interval_minutes * 60)
        self._stop_event = threading.Event()
        super().__init__(filename, mode="w", encoding=encoding, delay=False)
        self._initialize_rotation_state()
        self._worker = threading.Thread(
            target=self._run_periodic_truncate,
            name="pipeline-log-truncate",
            daemon=True,
        )
        self._worker.start()

    def _initialize_rotation_state(self) -> None:
        self._next_truncate_at = time.time() + self.interval_seconds

    def _run_periodic_truncate(self) -> None:
        while not self._stop_event.is_set():
            wait_seconds = max(1.0, self._next_truncate_at - time.time())
            if self._stop_event.wait(wait_seconds):
                return
            self._maybe_truncate()

    def _maybe_truncate(self) -> None:
        now = time.time()
        if now < self._next_truncate_at:
            return
        self.acquire()
        try:
            now = time.time()
            if now < self._next_truncate_at:
                return
            self._truncate_file_locked(now)
        finally:
            self.release()

    def _truncate_file_locked(self, now: float) -> None:
        if self.stream:
            self.stream.close()
            self.stream = None
        self.mode = "w"
        self.stream = self._open()
        self.mode = "a"
        self._next_truncate_at = now + self.interval_seconds

    def emit(self, record: logging.LogRecord) -> None:
        self._maybe_truncate()
        super().emit(record)

    def close(self) -> None:
        self._stop_event.set()
        super().close()


def configure_logging() -> None:
    global _CONFIGURED, _CONSOLE_HANDLER, _FILE_HANDLER
    if _CONFIGURED:
        return

    formatter = logging.Formatter(
        fmt="%(asctime)s [%(name)s] %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
    )

    if _env_bool("LOG_CONSOLE_ENABLED", True):
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.terminator = "\r\n"
        console_mode = _console_mode()
        if console_mode == "minimal":
            console_handler.addFilter(_MinimalConsoleFilter())
            console_handler.setFormatter(_MinimalConsoleFormatter(datefmt="%H:%M:%S"))
        elif console_mode == "normal":
            console_handler.addFilter(_NormalConsoleFilter())
            console_handler.setFormatter(formatter)
        else:
            console_handler.setFormatter(formatter)
        console_handler.setLevel(_log_level("LOG_CONSOLE_LEVEL", _log_level("LOG_LEVEL", logging.INFO)))
        setattr(console_handler, _HANDLER_TAG, True)
        _CONSOLE_HANDLER = console_handler

    if _env_bool("LOG_FILE_ENABLED", True):
        log_path = default_log_path()
        log_path.parent.mkdir(parents=True, exist_ok=True)
        _cleanup_expired_daily_logs(log_path.parent, log_path)
        rotate_minutes = _log_rotate_minutes()
        if rotate_minutes > 0:
            file_handler = _PeriodicTruncateFileHandler(
                log_path,
                interval_minutes=rotate_minutes,
                encoding="utf-8",
            )
        else:
            file_handler = RotatingFileHandler(
                log_path,
                maxBytes=max(1, _env_int("LOG_MAX_BYTES", 5 * 1024 * 1024)),
                backupCount=max(1, _env_int("LOG_BACKUP_COUNT", 3)),
                encoding="utf-8",
            )
        file_handler.setFormatter(formatter)
        file_mode = _file_mode()
        if file_mode == "minimal":
            file_handler.addFilter(_MinimalFileFilter())
        elif file_mode == "normal":
            file_handler.addFilter(_NormalFileFilter())
        file_handler.setLevel(_log_level("LOG_FILE_LEVEL", _log_level("LOG_LEVEL", logging.INFO)))
        setattr(file_handler, _HANDLER_TAG, True)
        _FILE_HANDLER = file_handler

    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    configure_logging()

    safe_name = name.strip() if name and name.strip() else "APP"
    full_name = f"{_APP_LOGGER_PREFIX}.{safe_name}"
    logger = logging.getLogger(full_name)
    logger.setLevel(_log_level("LOG_LEVEL", logging.INFO))
    logger.propagate = False

    if not getattr(logger, _LOGGER_TAG, False):
        if _CONSOLE_HANDLER is not None and _CONSOLE_HANDLER not in logger.handlers:
            logger.addHandler(_CONSOLE_HANDLER)
        if _FILE_HANDLER is not None and _FILE_HANDLER not in logger.handlers:
            logger.addHandler(_FILE_HANDLER)
        setattr(logger, _LOGGER_TAG, True)

    return logger
