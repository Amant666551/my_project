"""
api.py  –  FastAPI translation service

Key changes vs original:
  - /translate validates source_lang / target_lang before calling model
  - /health reports whether the model is actually loaded (not just "ok")
  - Startup event warms the model; 503 returned if it isn't ready yet
  - Proper HTTP 422 on bad input instead of silent 500
"""

import sys
import os
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, field_validator

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from translator import LightTranslator

logging.basicConfig(level=logging.INFO, format="%(asctime)s [API] %(message)s")
log = logging.getLogger(__name__)

SUPPORTED_LANGS = {"en", "zh", "ja", "ko", "fr", "de", "es", "ar"}

_translator: LightTranslator | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _translator
    log.info("Loading translation model …")
    _translator = LightTranslator()
    log.info("Model ready. API accepting requests.")
    yield
    log.info("Shutting down.")


app = FastAPI(title="Translation Service", lifespan=lifespan)


class TranslationRequest(BaseModel):
    text:        str
    source_lang: str = "en"
    target_lang: str = "zh"

    @field_validator("source_lang", "target_lang")
    @classmethod
    def validate_lang(cls, v: str) -> str:
        v = v.lower().strip()
        if v not in SUPPORTED_LANGS:
            raise ValueError(
                f"Unsupported language '{v}'. Supported: {sorted(SUPPORTED_LANGS)}"
            )
        return v

    @field_validator("text")
    @classmethod
    def validate_text(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("text must not be empty")
        if len(v) > 2000:
            raise ValueError("text exceeds 2000-character limit")
        return v


@app.post("/translate")
async def translate_text(req: TranslationRequest):
    if _translator is None:
        raise HTTPException(status_code=503, detail="Model not yet loaded, retry shortly.")

    try:
        result = _translator.translate(
            req.text,
            source_lang=req.source_lang,
            target_lang=req.target_lang,
        )
        return {
            "status":          "success",
            "original_text":   req.text,
            "translated_text": result,
            "source_lang":     req.source_lang,
            "target_lang":     req.target_lang,
        }
    except Exception as exc:
        log.exception("Translation failed")
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/health")
async def health():
    return {
        "status":       "ok" if _translator is not None else "loading",
        "model_loaded": _translator is not None,
    }
