from __future__ import annotations

import json
import re
from dataclasses import dataclass


@dataclass(frozen=True)
class SceneAnalysis:
    scene: str = "general"
    utterance_type: str = "statement"
    entity_focus: str = "none"
    register: str = "spoken"
    translation_hint: str = ""
    confidence: str = "medium"

    def summary_block(self) -> str:
        lines = ["Scene analysis from the disambiguation model:"]
        lines.append(f"- scene: {self.scene}")
        lines.append(f"- utterance_type: {self.utterance_type}")
        lines.append(f"- entity_focus: {self.entity_focus}")
        lines.append(f"- register: {self.register}")
        lines.append(f"- confidence: {self.confidence}")
        if self.translation_hint:
            lines.append(f"- translation_hint: {self.translation_hint}")
        return "\n".join(lines)

    def cache_key(self) -> tuple[str, str, str, str, str]:
        return (
            self.scene,
            self.utterance_type,
            self.entity_focus,
            self.register,
            self.translation_hint,
        )


def parse_scene_analysis(text: str) -> SceneAnalysis:
    payload = _extract_json_object(text)
    if not payload:
        return SceneAnalysis()

    try:
        data = json.loads(payload)
    except Exception:
        return SceneAnalysis()

    if not isinstance(data, dict):
        return SceneAnalysis()

    return SceneAnalysis(
        scene=_clean_label(data.get("scene", "general"), fallback="general"),
        utterance_type=_clean_label(data.get("utterance_type", "statement"), fallback="statement"),
        entity_focus=_clean_label(data.get("entity_focus", "none"), fallback="none"),
        register=_clean_label(data.get("register", "spoken"), fallback="spoken"),
        translation_hint=_clean_hint(data.get("translation_hint", "")),
        confidence=_clean_label(data.get("confidence", "medium"), fallback="medium"),
    )


def _extract_json_object(text: str) -> str:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)

    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start < 0 or end < start:
        return ""
    return cleaned[start : end + 1]


def _clean_label(value, *, fallback: str) -> str:
    text = str(value).strip().lower()
    if not text:
        return fallback
    text = re.sub(r"[^a-z0-9_ -]", "", text)
    return text or fallback


def _clean_hint(value) -> str:
    text = str(value).strip()
    text = re.sub(r"\s+", " ", text)
    return text[:240]
