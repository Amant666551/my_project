"""
record_voice.py - one-stop voice manager

Features:
- record a new voice sample into voice_samples/voice_XXX.wav
- create a Qwen TTS voice from that sample
- store created voices in voice_samples/voice_registry.json
- activate any stored voice by index

Default usage:
    python record_voice.py

Other usage:
    python record_voice.py --list
    python record_voice.py --activate 2
    python record_voice.py --duration 25
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import requests
import sounddevice as sd
import soundfile as sf
from dotenv import load_dotenv, set_key


load_dotenv()


BASE_DIR = Path(__file__).resolve().parent
ENV_PATH = BASE_DIR / ".env"
VOICE_DIR = BASE_DIR / "voice_samples"
REGISTRY_PATH = VOICE_DIR / "voice_registry.json"

DEFAULT_DURATION = 20
DEFAULT_SAMPLE_RATE = 22050
DEFAULT_REALTIME_TARGET_MODEL = "qwen3-tts-vc-realtime-2026-01-15"
DEFAULT_REALTIME_URL = "wss://dashscope.aliyuncs.com/api-ws/v1/realtime"
_env_target_model = os.getenv("QWEN_TTS_MODEL", "").strip()
DEFAULT_TARGET_MODEL = (
    _env_target_model
    if _env_target_model.startswith("qwen3-tts-vc-realtime")
    else DEFAULT_REALTIME_TARGET_MODEL
)
DEFAULT_PREFERRED_NAME = os.getenv("QWEN_TTS_PREFERRED_NAME", "myvoice")
DEFAULT_CUSTOMIZATION_URL = os.getenv(
    "QWEN_TTS_CUSTOMIZATION_URL",
    "https://dashscope.aliyuncs.com/api/v1/services/audio/tts/customization",
)
DEFAULT_REQUEST_TIMEOUT = (
    int(os.getenv("QWEN_TTS_CONNECT_TIMEOUT", "15")),
    int(os.getenv("QWEN_TTS_READ_TIMEOUT", "180")),
)


def _require_env_path() -> None:
    if not ENV_PATH.exists():
        raise FileNotFoundError(f".env not found: {ENV_PATH}")


def _load_registry() -> list[dict]:
    if not REGISTRY_PATH.exists():
        return []
    try:
        return json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        raise RuntimeError(f"Invalid registry JSON: {REGISTRY_PATH}")


def _save_registry(entries: list[dict]) -> None:
    VOICE_DIR.mkdir(parents=True, exist_ok=True)
    REGISTRY_PATH.write_text(
        json.dumps(entries, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _next_sample_path() -> Path:
    VOICE_DIR.mkdir(parents=True, exist_ok=True)
    existing = sorted(VOICE_DIR.glob("voice_*.wav"))
    max_index = 0
    for path in existing:
        stem = path.stem
        try:
            max_index = max(max_index, int(stem.split("_")[-1]))
        except ValueError:
            continue
    return VOICE_DIR / f"voice_{max_index + 1:03d}.wav"


def _guess_mime_type(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".wav":
        return "audio/wav"
    if suffix == ".mp3":
        return "audio/mpeg"
    if suffix == ".m4a":
        return "audio/mp4"
    raise ValueError(f"Unsupported audio format: {suffix}")


def _set_env_value(key: str, value: str) -> None:
    _require_env_path()
    set_key(str(ENV_PATH), key, value)
    os.environ[key] = value


def _is_realtime_vc_model(model_name: str) -> bool:
    normalized = (model_name or "").strip().lower()
    return normalized.startswith("qwen3-tts-vc-realtime")


def _sync_tts_env(
    *,
    voice_id: str,
    sample_path: str,
    target_model: str,
) -> None:
    _set_env_value("QWEN_TTS_MODEL", target_model)
    _set_env_value("QWEN_TTS_VOICE", voice_id)
    _set_env_value("QWEN_TTS_VOICE_SAMPLE", sample_path)
    _set_env_value("VOICE_SAMPLE", sample_path)
    if _is_realtime_vc_model(target_model):
        _set_env_value(
            "QWEN_TTS_URL",
            os.getenv("QWEN_TTS_URL", DEFAULT_REALTIME_URL) or DEFAULT_REALTIME_URL,
        )
        _set_env_value(
            "QWEN_TTS_SESSION_MODE",
            os.getenv("QWEN_TTS_SESSION_MODE", "commit") or "commit",
        )


def record_sample(
    duration: int,
    output: Path,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
) -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)

    print("\nVoice Sample Recorder")
    print(f"  Duration   : {duration}s")
    print(f"  Sample rate: {sample_rate} Hz")
    print(f"  Output     : {output}")
    print()
    print("Tips:")
    print("  - Speak naturally and clearly")
    print("  - Keep background noise low")
    print("  - Read a paragraph aloud")
    print()

    input("Press ENTER when ready...")

    for i in (3, 2, 1):
        print(f"Starting in {i}...", end="\r", flush=True)
        time.sleep(1)

    print("Recording... speak now.                 ")

    frames: list[np.ndarray] = []

    def callback(indata, frame_count, time_info, status):
        frames.append(indata.copy())
        level = float(np.abs(indata).mean())
        bar_len = min(int(level * 400), 40)
        bar = "#" * bar_len
        print(f"Level: {bar:<40}", end="\r", flush=True)

    with sd.InputStream(
        samplerate=sample_rate,
        channels=1,
        dtype="float32",
        blocksize=1024,
        callback=callback,
    ):
        sd.sleep(duration * 1000)

    print()
    print("Recording stopped.")

    audio = np.concatenate(frames, axis=0)
    peak = float(np.abs(audio).max())
    rms = float(np.sqrt(np.mean(audio ** 2)))
    sf.write(str(output), audio, sample_rate)

    print(f"Saved sample: {output}")
    print(f"Audio stats: peak={peak:.3f}, rms={rms:.4f}")
    return output


def create_qwen_voice_from_sample(
    audio_file: Path,
    target_model: str = DEFAULT_TARGET_MODEL,
    preferred_name: str = DEFAULT_PREFERRED_NAME,
) -> str:
    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        raise ValueError("DASHSCOPE_API_KEY is not set in .env")
    if not audio_file.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_file}")

    mime_type = _guess_mime_type(audio_file)
    base64_str = base64.b64encode(audio_file.read_bytes()).decode("utf-8")
    data_uri = f"data:{mime_type};base64,{base64_str}"

    payload = {
        "model": "qwen-voice-enrollment",
        "input": {
            "action": "create",
            "target_model": target_model,
            "preferred_name": preferred_name,
            "audio": {"data": data_uri},
        },
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    try:
        response = requests.post(
            DEFAULT_CUSTOMIZATION_URL,
            json=payload,
            headers=headers,
            timeout=DEFAULT_REQUEST_TIMEOUT,
        )
    except requests.exceptions.ReadTimeout as exc:
        raise RuntimeError(
            "Create voice request timed out. Try again later or increase "
            "QWEN_TTS_READ_TIMEOUT in .env."
        ) from exc
    except requests.exceptions.RequestException as exc:
        raise RuntimeError(f"Create voice request failed: {exc}") from exc

    print("status:", response.status_code)
    print(response.text)
    response.raise_for_status()

    try:
        return response.json()["output"]["voice"]
    except (KeyError, ValueError) as exc:
        raise RuntimeError(f"Failed to parse voice from response: {exc}") from exc


def activate_voice(entry_index: int) -> dict:
    entries = _load_registry()
    if not entries:
        raise RuntimeError("Voice registry is empty.")
    if entry_index < 1 or entry_index > len(entries):
        raise ValueError(f"Invalid voice index: {entry_index}")

    for i, entry in enumerate(entries, start=1):
        entry["active"] = i == entry_index

    chosen = entries[entry_index - 1]
    _save_registry(entries)

    _sync_tts_env(
        voice_id=chosen["qwen_tts_voice"],
        sample_path=chosen["sample_path"],
        target_model=chosen["target_model"],
    )

    return chosen


def delete_voices(entry_indices: list[int]) -> tuple[list[dict], dict | None]:
    entries = _load_registry()
    if not entries:
        raise RuntimeError("Voice registry is empty.")
    if not entry_indices:
        raise ValueError("No voice indices provided for deletion.")

    unique_indices = sorted(set(entry_indices))
    for entry_index in unique_indices:
        if entry_index < 1 or entry_index > len(entries):
            raise ValueError(f"Invalid voice index: {entry_index}")

    deleted_entries: list[dict] = []
    deleted_sample_paths: set[Path] = set()
    remaining_entries: list[dict] = []
    active_removed = False

    for idx, entry in enumerate(entries, start=1):
        if idx in unique_indices:
            deleted_entries.append(entry)
            if entry.get("active"):
                active_removed = True
            sample_rel = str(entry.get("sample_path", "")).strip()
            if sample_rel:
                deleted_sample_paths.add(BASE_DIR / sample_rel.replace("/", os.sep))
            continue
        remaining_entries.append(entry)

    for sample_path in sorted(deleted_sample_paths):
        try:
            if sample_path.exists():
                sample_path.unlink()
        except OSError as exc:
            raise RuntimeError(f"Failed to delete sample file: {sample_path}") from exc

    activated_entry: dict | None = None
    if remaining_entries:
        active_indices = [i for i, entry in enumerate(remaining_entries) if entry.get("active")]
        if active_removed or not active_indices:
            for entry in remaining_entries:
                entry["active"] = False
            remaining_entries[0]["active"] = True
            activated_entry = remaining_entries[0]
        else:
            activated_entry = remaining_entries[active_indices[0]]
    _save_registry(remaining_entries)

    if activated_entry is not None:
        _sync_tts_env(
            voice_id=activated_entry["qwen_tts_voice"],
            sample_path=activated_entry["sample_path"],
            target_model=activated_entry["target_model"],
        )

    return deleted_entries, activated_entry


def list_voices() -> None:
    entries = _load_registry()
    if not entries:
        print("No saved voices yet.")
        return

    print("\nSaved voices:")
    for i, entry in enumerate(entries, start=1):
        marker = "*" if entry.get("active") else " "
        print(
            f"[{i}] {marker} voice={entry['qwen_tts_voice']} | "
            f"sample={entry['sample_path']} | "
            f"model={entry['target_model']} | "
            f"created={entry['created_at']}"
        )
    print()


def register_voice(sample_path: Path, voice_id: str, target_model: str, preferred_name: str) -> dict:
    entries = _load_registry()
    for entry in entries:
        entry["active"] = False

    new_entry = {
        "sample_path": str(sample_path.relative_to(BASE_DIR)).replace("\\", "/"),
        "qwen_tts_voice": voice_id,
        "target_model": target_model,
        "preferred_name": preferred_name,
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "active": True,
    }
    entries.append(new_entry)
    _save_registry(entries)

    _sync_tts_env(
        voice_id=voice_id,
        sample_path=new_entry["sample_path"],
        target_model=target_model,
    )
    return new_entry


def record_create_and_activate(
    duration: int,
    sample_rate: int,
    target_model: str,
    preferred_name: str,
) -> None:
    sample_path = _next_sample_path()

    record_sample(duration=duration, output=sample_path, sample_rate=sample_rate)
    voice_id = create_qwen_voice_from_sample(
        audio_file=sample_path,
        target_model=target_model,
        preferred_name=preferred_name,
    )
    entry = register_voice(
        sample_path=sample_path,
        voice_id=voice_id,
        target_model=target_model,
        preferred_name=preferred_name,
    )

    print()
    print("Voice created and activated.")
    print(f"Sample : {entry['sample_path']}")
    print(f"Voice  : {entry['qwen_tts_voice']}")
    print(f"Model  : {entry['target_model']}")
    print("Updated .env keys:")
    print(f"  QWEN_TTS_MODEL={entry['target_model']}")
    print(f"  QWEN_TTS_VOICE={entry['qwen_tts_voice']}")
    print(f"  QWEN_TTS_VOICE_SAMPLE={entry['sample_path']}")
    print(f"  VOICE_SAMPLE={entry['sample_path']}")
    if _is_realtime_vc_model(entry["target_model"]):
        print(f"  QWEN_TTS_URL={os.getenv('QWEN_TTS_URL', DEFAULT_REALTIME_URL)}")
        print(f"  QWEN_TTS_SESSION_MODE={os.getenv('QWEN_TTS_SESSION_MODE', 'commit')}")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Record, create, list, and switch Qwen voices.")
    parser.add_argument("--duration", type=int, default=DEFAULT_DURATION, help="Recording duration in seconds.")
    parser.add_argument("--rate", type=int, default=DEFAULT_SAMPLE_RATE, help="Recording sample rate in Hz.")
    parser.add_argument("--list", action="store_true", help="List saved voices.")
    parser.add_argument("--activate", type=int, help="Activate a saved voice by 1-based index.")
    parser.add_argument(
        "--delete",
        type=int,
        nargs="+",
        help="Delete saved voices by 1-based index. Supports multiple indices, e.g. --delete 1 3 5.",
    )
    parser.add_argument("--list-devices", action="store_true", help="List available microphones and exit.")
    parser.add_argument(
        "--target-model",
        default=DEFAULT_TARGET_MODEL,
        help=f"Target model used when creating the voice. Default: {DEFAULT_TARGET_MODEL}",
    )
    parser.add_argument(
        "--preferred-name",
        default=DEFAULT_PREFERRED_NAME,
        help=f"Preferred voice name for Qwen voice enrollment. Default: {DEFAULT_PREFERRED_NAME}",
    )
    args = parser.parse_args()

    if args.list_devices:
        print(sd.query_devices())
        return

    if args.list:
        list_voices()
        return

    if args.activate is not None:
        chosen = activate_voice(args.activate)
        print()
        print("Activated voice:")
        print(f"  Voice  : {chosen['qwen_tts_voice']}")
        print(f"  Sample : {chosen['sample_path']}")
        print(f"  Model  : {chosen['target_model']}")
        return

    if args.delete is not None:
        deleted_entries, activated_entry = delete_voices(args.delete)
        print()
        print("Deleted voices:")
        for entry in deleted_entries:
            print(f"  Voice  : {entry['qwen_tts_voice']}")
            print(f"  Sample : {entry['sample_path']}")
            print(f"  Model  : {entry['target_model']}")
        if activated_entry is not None:
            print()
            print("Current active voice:")
            print(f"  Voice  : {activated_entry['qwen_tts_voice']}")
            print(f"  Sample : {activated_entry['sample_path']}")
            print(f"  Model  : {activated_entry['target_model']}")
        else:
            print()
            print("Voice registry is now empty.")
        return

    record_create_and_activate(
        duration=args.duration,
        sample_rate=args.rate,
        target_model=args.target_model,
        preferred_name=args.preferred_name,
    )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nOperation cancelled.")
        sys.exit(1)
    except Exception as exc:
        print(f"\nError: {exc}")
        sys.exit(1)
