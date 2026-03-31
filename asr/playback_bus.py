from __future__ import annotations

import os
import threading
import time

import numpy as np


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except ValueError:
        return default


class PlaybackBus:
    """Shared render-reference ring buffer for future AEC integration."""

    def __init__(self, sample_rate: int = 16000, max_seconds: float = 6.0):
        self.sample_rate = sample_rate
        self.capacity = max(1, int(sample_rate * max_seconds))
        self._buffer = np.zeros(self.capacity, dtype=np.float32)
        self._lock = threading.Lock()
        self._write_pos = 0
        self._total_written = 0
        self._last_activity = 0.0
        self._active_playbacks = 0

    def begin_playback(self) -> None:
        with self._lock:
            self._active_playbacks += 1
            self._last_activity = time.monotonic()

    def end_playback(self) -> None:
        with self._lock:
            self._active_playbacks = max(0, self._active_playbacks - 1)
            self._last_activity = time.monotonic()

    def is_recently_active(self, grace_ms: int = 250) -> bool:
        with self._lock:
            if self._active_playbacks > 0:
                return True
            last_activity = self._last_activity
        return (time.monotonic() - last_activity) * 1000.0 <= grace_ms

    def push_render_frame(self, frame: np.ndarray, sample_rate: int) -> None:
        samples = self._prepare_frame(frame, sample_rate)
        if samples.size == 0:
            return

        with self._lock:
            self._write_samples(samples)
            self._last_activity = time.monotonic()

    def get_recent_reference(self, num_samples: int, delay_samples: int = 0) -> np.ndarray:
        if num_samples <= 0:
            return np.empty(0, dtype=np.float32)

        out = np.zeros(num_samples, dtype=np.float32)
        with self._lock:
            available = min(self._total_written, self.capacity)
            if available <= 0:
                return out

            end_abs = max(0, self._total_written - max(0, delay_samples))
            start_abs = end_abs - num_samples
            earliest_abs = self._total_written - available

            fill_start_abs = max(start_abs, earliest_abs)
            if fill_start_abs >= end_abs:
                return out

            offset = fill_start_abs - start_abs
            length = end_abs - fill_start_abs
            out[offset : offset + length] = self._read_range(fill_start_abs, end_abs)
            return out

    def _prepare_frame(self, frame: np.ndarray, sample_rate: int) -> np.ndarray:
        samples = np.asarray(frame, dtype=np.float32)
        if samples.ndim == 2:
            samples = samples.mean(axis=1)
        elif samples.ndim > 2:
            samples = samples.reshape(-1)
        if samples.size == 0:
            return samples
        if sample_rate != self.sample_rate:
            samples = self._resample(samples, sample_rate, self.sample_rate)
        return np.clip(samples, -1.0, 1.0).astype(np.float32, copy=False)

    def _write_samples(self, samples: np.ndarray) -> None:
        count = len(samples)
        if count >= self.capacity:
            self._buffer[:] = samples[-self.capacity :]
            self._write_pos = 0
            self._total_written += count
            return

        end = self._write_pos + count
        if end <= self.capacity:
            self._buffer[self._write_pos : end] = samples
        else:
            first = self.capacity - self._write_pos
            self._buffer[self._write_pos :] = samples[:first]
            self._buffer[: end % self.capacity] = samples[first:]
        self._write_pos = end % self.capacity
        self._total_written += count

    def _read_range(self, start_abs: int, end_abs: int) -> np.ndarray:
        length = end_abs - start_abs
        if length <= 0:
            return np.empty(0, dtype=np.float32)

        start_mod = start_abs % self.capacity
        end_mod = end_abs % self.capacity
        if start_mod < end_mod:
            return self._buffer[start_mod:end_mod].copy()

        out = np.empty(length, dtype=np.float32)
        first = self.capacity - start_mod
        out[:first] = self._buffer[start_mod:]
        out[first:] = self._buffer[:end_mod]
        return out

    @staticmethod
    def _resample(samples: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
        if source_rate <= 0 or target_rate <= 0 or samples.size == 0:
            return samples.astype(np.float32, copy=False)
        target_len = max(1, int(round(samples.size * target_rate / source_rate)))
        source_x = np.linspace(0.0, 1.0, num=samples.size, endpoint=False)
        target_x = np.linspace(0.0, 1.0, num=target_len, endpoint=False)
        return np.interp(target_x, source_x, samples).astype(np.float32, copy=False)


PLAYBACK_REFERENCE_SAMPLE_RATE = int(os.getenv("AEC_SAMPLE_RATE", "16000"))
PLAYBACK_REFERENCE_SECONDS = _env_float("AEC_PLAYBACK_BUFFER_SEC", 6.0)

playback_bus = PlaybackBus(
    sample_rate=PLAYBACK_REFERENCE_SAMPLE_RATE,
    max_seconds=PLAYBACK_REFERENCE_SECONDS,
)
