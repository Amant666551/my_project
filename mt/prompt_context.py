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
    return text[: max(0, max_chars - 1)].rstrip() + "…"


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

    def build_prompt(self, text: str, source_lang: str, target_lang: str) -> str:
        if not self.enabled:
            return ""

        text = _normalize_space(text)
        if not text:
            return ""

        style_hints = self._style_hints(text)
        domain_hint = self._domain_hint(text)
        referent_hint = self._referent_hint(text)
        recent_context = self._recent_context_block()

        blocks: list[str] = []

        if style_hints:
            blocks.append("Speech-style hints:\n" + "\n".join(f"- {hint}" for hint in style_hints))
        if domain_hint:
            blocks.append(f"Domain hint:\n- {domain_hint}")
        if referent_hint:
            blocks.append(f"Disambiguation hint:\n- {referent_hint}")
        if recent_context:
            blocks.append(recent_context)

        return "\n\n".join(blocks)

    def _style_hints(self, text: str) -> list[str]:
        hints = [
            "This is live spoken dialogue, not polished written text.",
            "Keep the translation concise and natural for speech.",
        ]

        if any(token in text for token in ("嗯", "啊", "那个", "就是", "然后")):
            hints.append("If the utterance contains fillers, keep them light and do not over-interpret them.")
        if text.endswith(("吗", "呢", "？", "?")):
            hints.append("Preserve the interrogative tone.")
        if len(text) <= 8:
            hints.append("The utterance may be a short fragment; do not complete missing content unless necessary.")
        return hints

    def _domain_hint(self, text: str) -> str:
        academic_keywords = (
            "操作系统",
            "实验",
            "学院",
            "大学",
            "老师",
            "教授",
            "课程",
            "作业",
            "论文",
        )
        literature_keywords = (
            "小说",
            "作家",
            "评论",
            "史铁生",
            "文学",
        )
        technical_keywords = (
            "系统",
            "算法",
            "代码",
            "程序",
            "模型",
            "接口",
        )

        if any(keyword in text for keyword in academic_keywords):
            return "Interpret ambiguous terms in an academic or campus-discussion context when appropriate."
        if any(keyword in text for keyword in literature_keywords):
            return "Interpret ambiguous references in a literature, author, or commentary context when appropriate."
        if any(keyword in text for keyword in technical_keywords):
            return "Interpret ambiguous terms in a technical or computer-science context when appropriate."
        return ""

    def _referent_hint(self, text: str) -> str:
        if any(token in text for token in ("是谁", "你知道", "认识", "叫什么", "哪位")):
            return "If the utterance is asking about a person or institution, prefer named-entity interpretation over literal word-by-word translation."
        return ""

    def _recent_context_block(self) -> str:
        if not self.recent_turns:
            return ""

        lines = ["Recent dialogue context (for disambiguation only, do not re-translate it):"]
        for turn in list(self.recent_turns):
            lines.append(f"- source: {turn.source}")
            lines.append(f"  translation: {turn.translation}")
        return "\n".join(lines)
