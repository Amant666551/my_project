from __future__ import annotations

import os
import re
from collections import deque
from dataclasses import dataclass


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _trim_text(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[: max(0, max_chars - 1)].rstrip() + "..."


@dataclass(frozen=True)
class PromptTurn:
    source: str
    translation: str


class MTPromptContext:
    def __init__(self):
        self.enabled = _env_bool("MT_CONTEXT_ENABLED", True)
        self.max_turns = max(1, int(os.getenv("MT_CONTEXT_MAX_TURNS", "3")))
        self.max_chars_per_turn = max(12, int(os.getenv("MT_CONTEXT_MAX_CHARS_PER_TURN", "80")))
        self.recent_turns: deque[PromptTurn] = deque(maxlen=self.max_turns)

    def describe(self) -> str:
        return (
            "[MT prompt context] "
            f"enabled={self.enabled} | max_turns={self.max_turns} | "
            f"max_chars_per_turn={self.max_chars_per_turn} | cached_turns={len(self.recent_turns)}"
        )

    def observe_turn(self, source: str, translation: str) -> None:
        if not self.enabled:
            return

        source = _normalize_space(source)
        translation = _normalize_space(translation)
        if not source or not translation:
            return

        self.recent_turns.append(
            PromptTurn(
                source=_trim_text(source, self.max_chars_per_turn),
                translation=_trim_text(translation, self.max_chars_per_turn),
            )
        )

    def recent_context_block(self) -> str:
        if not self.enabled or not self.recent_turns:
            return ""

        lines = ["Recent dialogue context (for disambiguation only, do not re-translate it):"]
        for turn in list(self.recent_turns):
            lines.append(f"- source: {turn.source}")
            lines.append(f"  translation: {turn.translation}")
        return "\n".join(lines)

    def build_translation_prompt(self, scene_summary: str) -> str:
        blocks: list[str] = []
        if scene_summary:
            blocks.append(scene_summary)
        recent_context = self.recent_context_block()
        if recent_context:
            blocks.append(recent_context)
        return "\n\n".join(blocks)
