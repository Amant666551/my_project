import base64
import os
import pathlib

import requests
from dotenv import load_dotenv


load_dotenv()


TARGET_MODEL = os.getenv("QWEN_TTS_MODEL", "qwen3-tts-vc-2026-01-22")
PREFERRED_NAME = os.getenv("QWEN_TTS_PREFERRED_NAME", "myvoice")
AUDIO_FILE = os.getenv("QWEN_TTS_VOICE_SAMPLE", "voice_samples/my_voice.wav")
API_KEY = os.getenv("DASHSCOPE_API_KEY")
BASE_URL = os.getenv(
    "QWEN_TTS_CUSTOMIZATION_URL",
    "https://dashscope.aliyuncs.com/api/v1/services/audio/tts/customization",
)
REQUEST_TIMEOUT = (
    int(os.getenv("QWEN_TTS_CONNECT_TIMEOUT", "15")),
    int(os.getenv("QWEN_TTS_READ_TIMEOUT", "180")),
)


def _guess_mime_type(path: pathlib.Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".wav":
        return "audio/wav"
    if suffix == ".mp3":
        return "audio/mpeg"
    if suffix == ".m4a":
        return "audio/mp4"
    raise ValueError(f"Unsupported audio format: {suffix}")


def create_voice() -> str:
    if not API_KEY:
        raise ValueError("DASHSCOPE_API_KEY is not set in .env")

    file_path = pathlib.Path(AUDIO_FILE)
    if not file_path.exists():
        raise FileNotFoundError(f"Audio file not found: {file_path}")

    mime_type = _guess_mime_type(file_path)
    base64_str = base64.b64encode(file_path.read_bytes()).decode("utf-8")
    data_uri = f"data:{mime_type};base64,{base64_str}"

    payload = {
        "model": "qwen-voice-enrollment",
        "input": {
            "action": "create",
            "target_model": TARGET_MODEL,
            "preferred_name": PREFERRED_NAME,
            "audio": {"data": data_uri},
        },
    }
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }

    try:
        response = requests.post(
            BASE_URL,
            json=payload,
            headers=headers,
            timeout=REQUEST_TIMEOUT,
        )
    except requests.exceptions.ReadTimeout as exc:
        raise RuntimeError(
            "Create voice request timed out. Try again later, use a shorter/cleaner "
            "audio sample, or increase QWEN_TTS_READ_TIMEOUT in .env."
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


if __name__ == "__main__":
    voice = create_voice()
    print()
    print("VOICE =", voice)
    print()
    print("Put this into .env:")
    print(f"QWEN_TTS_VOICE={voice}")
