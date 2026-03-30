from __future__ import annotations

import os
import platform
import subprocess
import time
from pathlib import Path

import sounddevice as sd
import soundfile as sf

from playback_bus import playback_bus


PLAYER_CHUNK_SIZE = int(os.getenv("AEC_PLAYER_CHUNK_SIZE", "1024"))


def _log(logger, level: str, message: str, *args) -> None:
    if logger is not None and hasattr(logger, level):
        getattr(logger, level)(message, *args)
        return
    if args:
        try:
            message = message % args
        except Exception:
            message = f"{message} {' '.join(map(str, args))}"
    print(message)


def play_audio_file(path: str, logger=None) -> None:
    try:
        _play_with_sounddevice(path)
        return
    except Exception as exc:
        _log(
            logger,
            "warning",
            "Controlled playback failed (%s) - falling back to legacy player.",
            exc,
        )

    _play_with_legacy_player(path, logger=logger)


def _play_with_sounddevice(path: str) -> None:
    audio, sample_rate = sf.read(path, dtype="float32", always_2d=True)
    if audio.size == 0:
        raise ValueError("Audio file is empty.")

    channels = audio.shape[1]
    playback_bus.begin_playback()
    try:
        with sd.OutputStream(
            samplerate=sample_rate,
            channels=channels,
            dtype="float32",
            blocksize=PLAYER_CHUNK_SIZE,
        ) as stream:
            for start in range(0, len(audio), PLAYER_CHUNK_SIZE):
                chunk = audio[start : start + PLAYER_CHUNK_SIZE]
                stream.write(chunk)
                playback_bus.push_render_frame(chunk, sample_rate=sample_rate)
    finally:
        playback_bus.end_playback()


def _play_with_legacy_player(path: str, logger=None) -> None:
    try:
        import pygame

        pygame.mixer.init()
        pygame.mixer.music.load(path)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            time.sleep(0.05)
        pygame.mixer.music.stop()
        pygame.mixer.quit()
        return
    except Exception as exc:
        _log(logger, "warning", "pygame fallback failed (%s) - trying system player ...", exc)

    system = platform.system()
    try:
        if system == "Windows":
            os.startfile(path)
            time.sleep(max(2.0, Path(path).stat().st_size / 16_000))
        elif system == "Darwin":
            subprocess.run(["afplay", path], check=True)
        else:
            for player in ("mpg123", "mpg321", "ffplay", "aplay"):
                if subprocess.run(["which", player], capture_output=True).returncode == 0:
                    subprocess.run([player, "-q", path], check=True)
                    break
    except Exception as exc:
        _log(logger, "error", "System player failed: %s", exc)
