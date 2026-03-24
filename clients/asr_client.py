"""
clients/asr_client.py  –  ASR client wrapper (was empty in original)

Wraps the StreamingASR engine and exposes a simple iterator interface
so the orchestrator doesn't need to know about sounddevice / queues.

Usage
-----
    from clients.asr_client import ASRClient

    with ASRClient() as asr:
        for transcript in asr.stream():      # blocks; yields final sentences
            print(transcript)
"""

import queue
import sys

import numpy as np
import sounddevice as sd

from orchestrator import StreamingASR, SAMPLE_RATE, FRAME_SIZE, CHANNELS


class ASRClient:
    def __init__(self):
        self._asr   = StreamingASR()
        self._queue: queue.Queue[np.ndarray] = queue.Queue()
        self._stream = None

    # ── context manager ──────────────────────────────────────────────────────
    def __enter__(self):
        self._stream = sd.InputStream(
            samplerate=SAMPLE_RATE,
            blocksize=FRAME_SIZE,
            channels=CHANNELS,
            dtype="float32",
            callback=self._callback,
        )
        self._stream.start()
        return self

    def __exit__(self, *_):
        if self._stream:
            self._stream.stop()
            self._stream.close()

    def _callback(self, indata, frames, time_info, status):
        if status:
            print(status, file=sys.stderr)
        self._queue.put(indata[:, 0].copy())

    # ── public API ───────────────────────────────────────────────────────────
    def stream(self):
        """
        Generator that yields complete, final ASR transcripts.
        Blocks indefinitely; wrap in a thread or use inside a `with` block.
        """
        while True:
            chunk  = self._queue.get()
            result = self._asr.process_audio(chunk)
            if result:
                yield result
