from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _is_ascii_term(text: str) -> bool:
    return bool(text) and all(ord(ch) < 128 for ch in text)


def _contains_cjk(text: str) -> bool:
    return any("\u4e00" <= ch <= "\u9fff" for ch in text)


def _iter_cjk_spans(text: str) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    start = -1
    for index, ch in enumerate(text):
        if "\u4e00" <= ch <= "\u9fff":
            if start < 0:
                start = index
            continue
        if start >= 0:
            spans.append((start, index))
            start = -1
    if start >= 0:
        spans.append((start, len(text)))
    return spans


def _levenshtein_distance(left: tuple[str, ...], right: tuple[str, ...]) -> int:
    if left == right:
        return 0
    if not left:
        return len(right)
    if not right:
        return len(left)

    prev = list(range(len(right) + 1))
    for i, left_item in enumerate(left, start=1):
        curr = [i]
        for j, right_item in enumerate(right, start=1):
            cost = 0 if left_item == right_item else 1
            curr.append(
                min(
                    prev[j] + 1,
                    curr[j - 1] + 1,
                    prev[j - 1] + cost,
                )
            )
        prev = curr
    return prev[-1]


def _common_suffix_len(left: str, right: str) -> int:
    count = 0
    for left_ch, right_ch in zip(reversed(left), reversed(right)):
        if left_ch != right_ch:
            break
        count += 1
    return count


def _is_cjk_char(ch: str) -> bool:
    return "\u4e00" <= ch <= "\u9fff"


@dataclass(frozen=True)
class _HotwordRule:
    canonical: str
    alias: str
    pattern: re.Pattern[str]


@dataclass(frozen=True)
class _HotwordEntry:
    canonical: str
    category: str
    pinyin: tuple[str, ...]


class HotwordManager:
    def __init__(
        self,
        path: str | os.PathLike[str] | None = None,
        *,
        enabled: bool | None = None,
        max_replacements: int | None = None,
    ):
        base_dir = Path(__file__).resolve().parent
        self.path = Path(path) if path is not None else base_dir / "hotwords.json"
        self.enabled = _env_bool("HOTWORD_REWRITE_ENABLED", False) if enabled is None else enabled
        self.max_replacements = (
            max(1, int(os.getenv("HOTWORD_MAX_REPLACEMENTS", "2")))
            if max_replacements is None
            else max(1, max_replacements)
        )
        self.pinyin_enabled = _env_bool("HOTWORD_PINYIN_ENABLED", False)
        self.pinyin_min_score = float(os.getenv("HOTWORD_PINYIN_MIN_SCORE", "0.88"))
        self.pinyin_max_replacements = max(0, int(os.getenv("HOTWORD_PINYIN_MAX_REPLACEMENTS", "1")))
        self._rules: list[_HotwordRule] = []
        self._entry_count = 0
        self._pinyin_entries: list[_HotwordEntry] = []
        self._pinyin_entry_count = 0
        self._lazy_pinyin = None
        self._pinyin_available = False
        self._init_pinyin_backend()
        self._load()

    def _init_pinyin_backend(self) -> None:
        if not self.pinyin_enabled:
            return
        try:
            from pypinyin import Style, lazy_pinyin
        except Exception:
            return
        self._lazy_pinyin = lambda text: tuple(  # noqa: E731
            lazy_pinyin(text, style=Style.NORMAL, errors="ignore", strict=False)
        )
        self._pinyin_available = True

    def _load(self) -> None:
        if not self.enabled or not self.path.exists():
            return

        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except Exception as exc:
            print(f"[ASR hotwords] failed to load {self.path}: {exc}")
            return

        if not isinstance(payload, list):
            print(f"[ASR hotwords] ignored {self.path}: root must be a JSON list")
            return

        rules: list[_HotwordRule] = []
        pinyin_entries: list[_HotwordEntry] = []
        entry_count = 0
        for item in payload:
            if not isinstance(item, dict) or not item.get("enabled", True):
                continue
            canonical = str(item.get("canonical", "")).strip()
            if not canonical:
                continue
            category = str(item.get("category", "generic")).strip() or "generic"
            aliases = item.get("aliases", [])
            if not isinstance(aliases, list):
                continue

            seen_aliases: set[str] = set()
            local_rules: list[_HotwordRule] = []
            for alias_value in aliases:
                alias = str(alias_value).strip()
                if not alias or alias == canonical or alias in seen_aliases:
                    continue
                seen_aliases.add(alias)
                pattern = self._compile_pattern(alias)
                local_rules.append(
                    _HotwordRule(
                        canonical=canonical,
                        alias=alias,
                        pattern=pattern,
                    )
                )

            if local_rules:
                entry_count += 1
                rules.extend(local_rules)
            elif canonical:
                entry_count += 1

            pinyin = self._entry_pinyin(canonical, item.get("pinyin"))
            if pinyin:
                pinyin_entries.append(
                    _HotwordEntry(
                        canonical=canonical,
                        category=category,
                        pinyin=pinyin,
                    )
                )

        rules.sort(key=lambda rule: len(rule.alias), reverse=True)
        pinyin_entries.sort(key=lambda entry: len(entry.canonical), reverse=True)
        self._rules = rules
        self._entry_count = entry_count
        self._pinyin_entries = pinyin_entries
        self._pinyin_entry_count = len(pinyin_entries)

    def _entry_pinyin(self, canonical: str, raw_pinyin) -> tuple[str, ...]:
        if not self.pinyin_enabled or not self._pinyin_available or not _contains_cjk(canonical):
            return ()
        if isinstance(raw_pinyin, list):
            cleaned = tuple(str(item).strip().lower() for item in raw_pinyin if str(item).strip())
            if cleaned:
                return cleaned
        if self._lazy_pinyin is None:
            return ()
        return tuple(part.lower() for part in self._lazy_pinyin(canonical) if part)

    @staticmethod
    def _compile_pattern(alias: str) -> re.Pattern[str]:
        escaped = re.escape(alias)
        if _is_ascii_term(alias):
            return re.compile(rf"(?<![A-Za-z0-9]){escaped}(?![A-Za-z0-9])", re.IGNORECASE)
        return re.compile(escaped)

    def rewrite(self, text: str) -> tuple[str, list[tuple[str, str]]]:
        if not self.enabled or not self._rules:
            return self._rewrite_by_pinyin(text, [], self.max_replacements)

        rewritten = text
        hits: list[tuple[str, str]] = []
        replacement_count = 0

        for rule in self._rules:
            if replacement_count >= self.max_replacements:
                break

            rewritten_next, count = rule.pattern.subn(rule.canonical, rewritten, count=1)
            if count <= 0 or rewritten_next == rewritten:
                continue

            rewritten = rewritten_next
            hits.append((rule.alias, rule.canonical))
            replacement_count += 1

        return self._rewrite_by_pinyin(rewritten, hits, self.max_replacements - replacement_count)

    def _rewrite_by_pinyin(
        self,
        text: str,
        hits: list[tuple[str, str]],
        remaining_replacements: int,
    ) -> tuple[str, list[tuple[str, str]]]:
        if (
            not self.enabled
            or not self.pinyin_enabled
            or not self._pinyin_available
            or remaining_replacements <= 0
            or self.pinyin_max_replacements <= 0
            or not self._pinyin_entries
        ):
            return text, hits

        rewritten = text
        pinyin_replacements = 0
        while remaining_replacements > 0 and pinyin_replacements < self.pinyin_max_replacements:
            candidate = self._best_pinyin_candidate(rewritten)
            if candidate is None:
                break

            start, end, matched_text, canonical, _score = candidate
            rewritten = rewritten[:start] + canonical + rewritten[end:]
            hits.append((matched_text, canonical))
            remaining_replacements -= 1
            pinyin_replacements += 1

        return rewritten, hits

    def _best_pinyin_candidate(
        self,
        text: str,
    ) -> tuple[int, int, str, str, float] | None:
        best: tuple[int, int, str, str, float] | None = None
        for span_start, span_end in _iter_cjk_spans(text):
            span = text[span_start:span_end]
            for entry in self._pinyin_entries:
                canonical_len = len(entry.canonical)
                min_len = max(2, canonical_len - 1)
                max_len = min(len(span), canonical_len + 1)
                for candidate_len in range(min_len, max_len + 1):
                    for local_start in range(0, len(span) - candidate_len + 1):
                        candidate_text = span[local_start : local_start + candidate_len]
                        if candidate_text == entry.canonical:
                            continue
                        if entry.canonical in candidate_text or candidate_text in entry.canonical:
                            continue
                        if not self._is_valid_cjk_boundary(text, span_start + local_start, span_start + local_start + candidate_len):
                            continue
                        score = self._pinyin_score(candidate_text, entry)
                        if score < self.pinyin_min_score:
                            continue
                        absolute_start = span_start + local_start
                        absolute_end = absolute_start + candidate_len
                        if best is None or score > best[4] or (
                            score == best[4] and len(entry.canonical) > len(best[3])
                        ):
                            best = (
                                absolute_start,
                                absolute_end,
                                candidate_text,
                                entry.canonical,
                                score,
                            )
        return best

    @staticmethod
    def _is_valid_cjk_boundary(text: str, start: int, end: int) -> bool:
        prev_char = text[start - 1] if start > 0 else ""
        next_char = text[end] if end < len(text) else ""

        prev_is_cjk = bool(prev_char) and _is_cjk_char(prev_char)
        next_is_cjk = bool(next_char) and _is_cjk_char(next_char)

        # Avoid rewriting the middle of a longer Chinese phrase.
        return not prev_is_cjk and not next_is_cjk

    def _pinyin_score(self, candidate_text: str, entry: _HotwordEntry) -> float:
        if self._lazy_pinyin is None:
            return 0.0
        candidate_pinyin = tuple(part.lower() for part in self._lazy_pinyin(candidate_text) if part)
        if not candidate_pinyin:
            return 0.0

        max_len = max(len(candidate_pinyin), len(entry.pinyin))
        if max_len <= 0:
            return 0.0

        edit_distance = _levenshtein_distance(candidate_pinyin, entry.pinyin)
        pinyin_similarity = 1.0 - (edit_distance / max_len)
        char_similarity = SequenceMatcher(None, candidate_text, entry.canonical).ratio()
        suffix_len = _common_suffix_len(candidate_text, entry.canonical)
        suffix_bonus = 0.08 if suffix_len >= 2 else 0.0
        score = (pinyin_similarity * 0.75) + (char_similarity * 0.25) + suffix_bonus
        return min(score, 1.0)

    def describe(self) -> str:
        return (
            "[ASR hotwords] "
            f"enabled={self.enabled} | entries={self._entry_count} | aliases={len(self._rules)} | "
            f"pinyin_enabled={self.pinyin_enabled} | pinyin_entries={self._pinyin_entry_count} | "
            f"max_replacements={self.max_replacements} | file={self.path.name}"
        )
