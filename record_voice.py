"""
record_voice.py  –  Voice sample recorder for TTS cloning

Records a clean reference clip of your voice and saves it to
voice_samples/my_voice.wav, which main.py reads for voice cloning.

Usage
-----
    python record_voice.py                  # default 20 s
    python record_voice.py --duration 30    # record 30 s
    python record_voice.py --output voice_samples/custom.wav

Tips for a good reference clip
--------------------------------
- Find a quiet room — background noise degrades cloning quality significantly
- Speak naturally at a comfortable pace; avoid rushing or whispering
- 15–30 seconds is the sweet spot; very short clips (<6 s) reduce quality
- Read aloud naturally, e.g. a paragraph from a book or article
- Keep the microphone 15–30 cm from your mouth
- Watch the level meter: aim to stay in the green/yellow range

What each model needs
----------------------
  XTTS-v2     6 s minimum, 15-30 s recommended
  OpenVoice   5 s minimum, 15-20 s recommended
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import sounddevice as sd
import soundfile as sf

DEFAULT_DURATION   = 20        # seconds
DEFAULT_SAMPLE_RATE = 22050    # Hz — XTTS and OpenVoice both prefer 22050
DEFAULT_OUTPUT     = "voice_samples/my_voice.wav"


def record(duration: int, output: str, sample_rate: int = DEFAULT_SAMPLE_RATE) -> None:
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"\n🎙️  Voice Sample Recorder")
    print(f"   Duration   : {duration} s")
    print(f"   Sample rate: {sample_rate} Hz")
    print(f"   Output     : {output_path.resolve()}")
    print()
    print("Tips:")
    print("  - Speak naturally and clearly")
    print("  - Read a paragraph aloud — variety of sounds helps cloning")
    print("  - Keep background noise to a minimum")
    print()

    input("Press ENTER when you're ready to start recording ...")
    print()

    # Countdown
    for i in (3, 2, 1):
        print(f"  Starting in {i} ...", end="\r", flush=True)
        import time; time.sleep(1)

    print("  ● RECORDING — speak now!              ")
    print()

    # Show a live level meter while recording
    frames = []
    block_size = 1024

    def callback(indata, frame_count, time_info, status):
        frames.append(indata.copy())
        # Simple ASCII level meter
        level = np.abs(indata).mean()
        bar_len = int(level * 400)
        bar = "█" * min(bar_len, 40)
        color = "\033[32m" if bar_len < 25 else "\033[33m" if bar_len < 38 else "\033[31m"
        print(f"  Level: {color}{bar:<40}\033[0m", end="\r", flush=True)

    with sd.InputStream(
        samplerate=sample_rate,
        channels=1,
        dtype="float32",
        blocksize=block_size,
        callback=callback,
    ):
        sd.sleep(duration * 1000)

    print()
    print()
    print("  ■ Recording stopped.")

    # Concatenate and save
    audio = np.concatenate(frames, axis=0)

    # Warn if the recording was too quiet
    peak = np.abs(audio).max()
    rms  = np.sqrt(np.mean(audio ** 2))
    if peak < 0.05:
        print("\n⚠️  Warning: recording is very quiet (peak={:.3f}).".format(peak))
        print("   Check your microphone and try again.")
    elif rms < 0.01:
        print("\n⚠️  Warning: low average volume (RMS={:.4f}).".format(rms))
        print("   Try speaking louder or moving closer to the microphone.")
    else:
        print(f"   Audio stats: peak={peak:.3f}, RMS={rms:.4f}  ✅ looks good")

    sf.write(str(output_path), audio, sample_rate)

    size_kb = output_path.stat().st_size / 1024
    print(f"\n✅ Saved: {output_path.resolve()}  ({size_kb:.1f} KB, {duration} s)")
    print()
    print("Next step: open main.py and set:")
    print(f'   VOICE_SAMPLE = "{output}"')
    print('   TTS_BACKEND  = "xtts"   # or "openvoice"')
    print()
    print("Then run your pipeline normally:")
    print("   python orchestrator.py")
    print()


def list_devices() -> None:
    print("\nAvailable audio input devices:")
    print(sd.query_devices())
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Record a voice reference clip for TTS cloning.")
    parser.add_argument("--duration", type=int,  default=DEFAULT_DURATION,
                        help=f"Recording duration in seconds (default: {DEFAULT_DURATION})")
    parser.add_argument("--output",   type=str,  default=DEFAULT_OUTPUT,
                        help=f"Output WAV path (default: {DEFAULT_OUTPUT})")
    parser.add_argument("--rate",     type=int,  default=DEFAULT_SAMPLE_RATE,
                        help=f"Sample rate in Hz (default: {DEFAULT_SAMPLE_RATE})")
    parser.add_argument("--list-devices", action="store_true",
                        help="List available microphones and exit")
    args = parser.parse_args()

    if args.list_devices:
        list_devices()
        sys.exit(0)

    try:
        record(duration=args.duration, output=args.output, sample_rate=args.rate)
    except KeyboardInterrupt:
        print("\n\nRecording cancelled.")
        sys.exit(1)
    except Exception as exc:
        print(f"\n❌ Error: {exc}")
        sys.exit(1)