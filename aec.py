from __future__ import annotations

import os

import numpy as np

from playback_bus import playback_bus


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


class _PassthroughAECBackend:
    name = "passthrough"

    def process_capture(self, mic_frame: np.ndarray, render_frame: np.ndarray) -> np.ndarray:
        _ = render_frame
        return mic_frame


class _PyaecBackend:
    name = "pyaec"

    def __init__(self, sample_rate: int, frame_size: int):
        import pyaec

        self.sample_rate = sample_rate
        self.capture_frame_size = frame_size
        self.process_frame_size = max(1, _env_int("AEC_FRAME_SIZE", 160))
        self.filter_length = max(self.process_frame_size, _env_int("AEC_FILTER_LENGTH", 1600))
        self._engine = self._create_engine(pyaec)
        self._process_method_name = self._detect_process_method(self._engine)

    def _create_engine(self, module):
        candidate = getattr(module, "Aec", None)
        if candidate is None:
            raise RuntimeError("pyaec module does not expose class 'Aec'.")

        return candidate(
            self.process_frame_size,
            self.filter_length,
            self.sample_rate,
            True,
        )

    @staticmethod
    def _detect_process_method(engine) -> str:
        for method_name in ("cancel_echo", "process", "process_frame"):
            method = getattr(engine, method_name, None)
            if callable(method):
                return method_name
        raise RuntimeError(
            "pyaec engine does not expose a supported processing method. "
            "Expected 'process', 'cancel_echo', or 'process_frame'."
        )

    def process_capture(self, mic_frame: np.ndarray, render_frame: np.ndarray) -> np.ndarray:
        mic = self._to_int16(mic_frame)
        render = self._to_int16(render_frame)
        target_len = len(mic)
        if target_len == 0:
            return mic_frame

        chunk_size = self.process_frame_size
        padded_len = ((target_len + chunk_size - 1) // chunk_size) * chunk_size
        if padded_len != target_len:
            mic = np.pad(mic, (0, padded_len - target_len))
            render = np.pad(render, (0, padded_len - target_len))

        outputs: list[np.ndarray] = []
        for start in range(0, padded_len, chunk_size):
            mic_chunk = mic[start : start + chunk_size]
            render_chunk = render[start : start + chunk_size]
            outputs.append(self._process_chunk(mic_chunk, render_chunk))

        merged = np.concatenate(outputs)[:target_len]
        return (merged.astype(np.float32) / 32768.0).astype(np.float32, copy=False)

    def _process_chunk(self, mic_chunk: np.ndarray, render_chunk: np.ndarray) -> np.ndarray:
        processed = getattr(self._engine, self._process_method_name)(mic_chunk, render_chunk)
        return np.asarray(processed, dtype=np.int16).reshape(-1)

    @staticmethod
    def _to_int16(frame: np.ndarray) -> np.ndarray:
        clipped = np.clip(np.asarray(frame, dtype=np.float32).reshape(-1), -1.0, 1.0)
        return (clipped * 32767.0).astype(np.int16)


class EchoCanceller:
    """AEC wrapper with optional real backend support."""

    def __init__(self, frame_size: int, sample_rate: int):
        self.frame_size = frame_size
        self.sample_rate = sample_rate
        self.enabled = _env_bool("ENABLE_AEC", False)
        self.backend_name = os.getenv("AEC_BACKEND", "pyaec").strip().lower() or "pyaec"
        self.delay_ms = _env_int("AEC_DELAY_MS", 120)
        self.delay_samples = max(0, int(self.delay_ms * self.sample_rate / 1000))
        self.bypass_when_no_render = _env_bool("AEC_BYPASS_WHEN_NO_RENDER", True)
        self._backend = self._load_backend()

    def _load_backend(self):
        if not self.enabled:
            return _PassthroughAECBackend()

        if self.backend_name in {"none", "passthrough", "stub"}:
            print("[AEC] enabled=True | backend=passthrough | note=ready for future backend")
            return _PassthroughAECBackend()

        if self.backend_name in {"auto", "pyaec"}:
            try:
                backend = _PyaecBackend(
                    sample_rate=self.sample_rate,
                    frame_size=self.frame_size,
                )
                print(
                    "[AEC] backend=pyaec initialized | "
                    f"sample_rate={self.sample_rate} | frame_size={backend.process_frame_size} | "
                    f"filter_length={backend.filter_length}"
                )
                return backend
            except ImportError:
                print(
                    "[AEC] backend=pyaec requested, but package 'pyaec' is not installed. "
                    "Falling back to passthrough."
                )
                return _PassthroughAECBackend()
            except Exception as exc:
                print(f"[AEC] pyaec initialization failed: {exc}. Falling back to passthrough.")
                return _PassthroughAECBackend()

        if self.backend_name == "webrtc":
            print(
                "[AEC] backend=webrtc requested, but no WebRTC binding is installed yet. "
                "Falling back to passthrough."
            )
            return _PassthroughAECBackend()

        print(f"[AEC] unknown backend '{self.backend_name}', falling back to passthrough.")
        return _PassthroughAECBackend()

    def process_capture(self, mic_frame: np.ndarray) -> np.ndarray:
        frame = np.asarray(mic_frame, dtype=np.float32).reshape(-1)
        if frame.size == 0:
            return frame

        if not self.enabled:
            return frame

        if self.bypass_when_no_render and not playback_bus.is_recently_active():
            return frame

        render_ref = playback_bus.get_recent_reference(
            num_samples=len(frame),
            delay_samples=self.delay_samples,
        )
        return np.asarray(
            self._backend.process_capture(frame, render_ref),
            dtype=np.float32,
        )

    def describe(self) -> str:
        parts = [
            "[AEC]",
            f"enabled={self.enabled}",
            f"requested={self.backend_name}",
            f"active={self._backend.name}",
            f"sample_rate={self.sample_rate}",
            f"frame_size={self.frame_size}",
            f"delay_ms={self.delay_ms}",
        ]
        process_frame_size = getattr(self._backend, "process_frame_size", None)
        if process_frame_size is not None:
            parts.append(f"backend_frame_size={process_frame_size}")
        filter_length = getattr(self._backend, "filter_length", None)
        if filter_length is not None:
            parts.append(f"filter_length={filter_length}")
        return " | ".join(parts)
