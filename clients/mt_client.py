"""
clients/mt_client.py  –  Optimized MT client

Key changes vs original:
  - Calls the FastAPI /translate HTTP endpoint directly instead of going
    through two Redis pub/sub hops (saves ~50–200 ms + serialization overhead)
  - Retries once on transient connection errors
  - Returns None (not empty string) on failure so callers can distinguish
    "translated to empty" from "translation failed"
"""

import requests


class MTClient:
    def __init__(self, base_url: str = "http://127.0.0.1:8000"):
        self.base_url = base_url.rstrip("/")
        self._session = requests.Session()   # keep-alive connection pooling

    def translate(
        self,
        text: str,
        source_lang: str = "en",
        target_lang: str = "zh",
        timeout: float = 10.0,
    ) -> str | None:
        """
        Translate *text* via the FastAPI service.
        Returns the translated string, or None on error / timeout.
        """
        if not text or not text.strip():
            return None

        payload = {
            "text":        text,
            "source_lang": source_lang,
            "target_lang": target_lang,
        }

        for attempt in range(2):   # one retry on transient failure
            try:
                resp = self._session.post(
                    f"{self.base_url}/translate",
                    json=payload,
                    timeout=timeout,
                )
                resp.raise_for_status()
                return resp.json().get("translated_text", "").strip() or None
            except requests.exceptions.Timeout:
                print(f"⚠️  [MTClient] Timeout on attempt {attempt + 1}")
                if attempt == 0:
                    continue
                return None
            except requests.exceptions.ConnectionError:
                print("❌ [MTClient] Cannot reach translation service. Is api.py running?")
                return None
            except Exception as exc:
                print(f"❌ [MTClient] Unexpected error: {exc}")
                return None

        return None
