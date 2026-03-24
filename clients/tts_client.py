"""
clients/tts_client.py  –  Fixed TTS client

Root cause of the timeout
-------------------------
The previous TTSClient published to channel:mt_to_tts and then blocked
waiting for audio_base64 bytes on channel:tts_to_speaker.
main.py NEVER published to channel:tts_to_speaker — it only played
audio locally. So TTSClient always timed out, by design.

Fix
---
On the same machine: don't use TTSClient at all. Import speak() directly:

    from main import speak
    speak("你好", lang="zh")   # no Redis, no timeout, no network

That is what orchestrator.py now does.

This file is kept for the case where the TTS speaker runs on a
DIFFERENT machine. In that mode the orchestrator publishes a message
and does NOT wait for a reply (fire-and-forget). speak() returns
immediately after publishing.
"""

import json
import logging
import sys

import redis

log = logging.getLogger("tts_client")
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [TTSClient] %(levelname)-8s %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)


class TTSClient:
    """
    Fire-and-forget Redis client for a *remote* TTS speaker.

    If the TTS speaker is on the same machine, import speak() from main.py
    directly — this class is unnecessary and only adds Redis round-trip
    latency.
    """

    def __init__(self, redis_host: str = "127.0.0.1", redis_port: int = 6379):
        self._redis = redis.Redis(
            host=redis_host, port=redis_port, decode_responses=True
        )
        try:
            self._redis.ping()
            log.info("Connected to Redis (%s:%d)", redis_host, redis_port)
        except Exception as exc:
            raise ConnectionError(
                f"Redis unavailable at {redis_host}:{redis_port}: {exc}"
            ) from exc

        self.pub_channel = "channel:mt_to_tts"

    def speak(self, text: str, lang: str = "en", **_ignored) -> None:
        """
        Publish a TTS request to the remote speaker service.

        This is fire-and-forget — it returns as soon as the message is
        published. There is no reply channel; do not wait for audio bytes.

        Parameters
        ----------
        text : str   Text to synthesise.
        lang : str   Language code ("zh", "en", …).
        """
        if not text or not text.strip():
            log.debug("speak() called with empty text — skipping.")
            return

        payload = json.dumps({"text": text.strip(), "lang": lang})
        recipients = self._redis.publish(self.pub_channel, payload)
        log.info(
            "Published to '%s' (%d subscriber(s)): %r",
            self.pub_channel, recipients, text[:50],
        )

        if recipients == 0:
            log.warning(
                "No subscribers on '%s'. "
                "Is main.py running on the speaker machine?",
                self.pub_channel,
            )