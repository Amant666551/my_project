from __future__ import annotations

import argparse
import html
import json
import os
import re
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import requests


SOURCE_TIMEOUT_SEC = int(os.getenv("HOTWORD_SOURCE_TIMEOUT_SEC", "12"))
CANDIDATE_CONTEXT_LIMIT = max(1, int(os.getenv("HOTWORD_CANDIDATE_CONTEXT_LIMIT", "3")))
AUTO_PROMOTE_THRESHOLD = float(os.getenv("HOTWORD_AUTO_PROMOTE_THRESHOLD", "0.93"))

_CJK_RE = re.compile(r"[\u4e00-\u9fff]")
_ASCII_TERM_RE = re.compile(r"\b[A-Za-z][A-Za-z0-9_-]{2,}\b")
_CJK_TERM_RE = re.compile(r"[\u4e00-\u9fff]{2,10}")
_ORG_TERM_RE = re.compile(
    r"[\u4e00-\u9fffA-Za-z0-9]{2,24}(?:学院|大学|学校|实验室|研究院|研究所|中心|课程|系统|平台|项目|专业|图书馆)"
)
_HTML_BLOCK_RE = re.compile(r"<(title|h1|h2|h3|li|p|td|th|code)[^>]*>(.*?)</\1>", re.IGNORECASE | re.DOTALL)
_HTML_TAG_RE = re.compile(r"<[^>]+>")
_HTML_SCRIPT_STYLE_RE = re.compile(r"<(script|style)[^>]*>.*?</\1>", re.IGNORECASE | re.DOTALL)

_COMMON_STOP_TERMS = {
    "今天",
    "你好",
    "可以",
    "知道",
    "一下",
    "因为",
    "所以",
    "这个",
    "那个",
    "我们",
    "你们",
    "他们",
    "系统",
}
_ASCII_STOP_TERMS = {
    "http",
    "https",
    "www",
    "html",
    "body",
    "class",
    "style",
    "title",
}


def _contains_cjk(text: str) -> bool:
    return bool(_CJK_RE.search(text))


def _normalize_text(text: str) -> str:
    text = html.unescape(text)
    text = text.replace("\u3000", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _safe_read_json(path: Path, default):
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def _lazy_pinyin(text: str) -> list[str]:
    try:
        from pypinyin import Style, lazy_pinyin
    except Exception:
        return []
    return lazy_pinyin(text, style=Style.NORMAL, errors="ignore", strict=False)


@dataclass
class _SourceBlock:
    text: str
    weight: float
    tag: str


@dataclass
class _CandidateSignal:
    term: str
    category: str
    score: float
    context: str
    source_ref: str


class HotwordLearner:
    def __init__(
        self,
        *,
        base_dir: Path | None = None,
        sources_path: Path | None = None,
        candidates_path: Path | None = None,
        hotwords_path: Path | None = None,
    ):
        self.base_dir = base_dir or Path(__file__).resolve().parent.parent
        self.sources_path = sources_path or (Path(__file__).resolve().parent / "hotword_sources.json")
        self.candidates_path = candidates_path or (Path(__file__).resolve().parent / "hotword_candidates.json")
        self.hotwords_path = hotwords_path or (Path(__file__).resolve().parent / "hotwords.json")
        self.sources = self._load_sources()
        self.existing_hotwords = self._load_existing_hotwords()
        self.candidates = self._load_candidates()

    def _load_sources(self) -> list[dict]:
        payload = _safe_read_json(self.sources_path, [])
        if not isinstance(payload, list):
            return []
        return [item for item in payload if isinstance(item, dict) and item.get("enabled", True)]

    def _load_existing_hotwords(self) -> set[str]:
        payload = _safe_read_json(self.hotwords_path, [])
        terms: set[str] = set()
        if not isinstance(payload, list):
            return terms
        for item in payload:
            if not isinstance(item, dict) or not item.get("enabled", True):
                continue
            canonical = str(item.get("canonical", "")).strip()
            if canonical:
                terms.add(canonical)
            aliases = item.get("aliases", [])
            if isinstance(aliases, list):
                for alias in aliases:
                    alias_text = str(alias).strip()
                    if alias_text:
                        terms.add(alias_text)
        return terms

    def _load_candidates(self) -> dict[str, dict]:
        payload = _safe_read_json(self.candidates_path, [])
        if not isinstance(payload, list):
            return {}
        candidates: dict[str, dict] = {}
        for item in payload:
            if not isinstance(item, dict):
                continue
            term = str(item.get("term", "")).strip()
            if not term:
                continue
            candidates[term] = item
        return candidates

    def run(self) -> tuple[int, int]:
        if not self.sources:
            print(f"[Hotword learner] no enabled sources found in {self.sources_path.name}")
            return 0, 0

        discovered: dict[str, list[_CandidateSignal]] = defaultdict(list)
        processed_sources = 0
        for source in self.sources:
            processed_sources += 1
            try:
                source_ref, raw_text = self._load_source_text(source)
            except Exception as exc:
                print(f"[Hotword learner] failed source={source.get('name', 'unknown')}: {exc}")
                continue
            for signal in self._extract_signals(source, source_ref, raw_text):
                if signal.term in self.existing_hotwords:
                    continue
                discovered[signal.term].append(signal)

        updated = self._merge_discovered(discovered)
        self._write_candidates()
        print(
            "[Hotword learner] "
            f"processed_sources={processed_sources} | discovered_terms={len(discovered)} | updated_candidates={updated}"
        )
        return processed_sources, updated

    def _load_source_text(self, source: dict) -> tuple[str, str]:
        path_value = str(source.get("path", "")).strip()
        url_value = str(source.get("url", "")).strip()
        if path_value:
            target = (self.base_dir / path_value).resolve()
            text = target.read_text(encoding="utf-8", errors="ignore")
            return str(target), text
        if url_value:
            resp = requests.get(url_value, timeout=SOURCE_TIMEOUT_SEC, headers={"User-Agent": "hotword-learner/1.0"})
            resp.raise_for_status()
            return url_value, resp.text
        raise ValueError("source must define either 'path' or 'url'")

    def _extract_signals(self, source: dict, source_ref: str, raw_text: str) -> Iterable[_CandidateSignal]:
        category = str(source.get("category", "generic")).strip() or "generic"
        suffix_hint = category in {"organization", "project_term"}
        if raw_text.lstrip().startswith("<"):
            blocks = self._extract_html_blocks(raw_text)
        else:
            blocks = self._extract_text_blocks(raw_text)

        for block in blocks:
            for term, score in self._extract_terms_from_block(block, category=category, suffix_hint=suffix_hint):
                yield _CandidateSignal(
                    term=term,
                    category=category,
                    score=score,
                    context=block.text[:120],
                    source_ref=source_ref,
                )

    def _extract_html_blocks(self, html_text: str) -> list[_SourceBlock]:
        cleaned = _HTML_SCRIPT_STYLE_RE.sub(" ", html_text)
        blocks: list[_SourceBlock] = []
        for tag, inner in _HTML_BLOCK_RE.findall(cleaned):
            text = _normalize_text(_HTML_TAG_RE.sub(" ", inner))
            if not text:
                continue
            weight = {
                "title": 1.0,
                "h1": 0.95,
                "h2": 0.9,
                "h3": 0.85,
                "code": 0.85,
                "li": 0.75,
                "p": 0.55,
                "td": 0.45,
                "th": 0.5,
            }.get(tag.lower(), 0.4)
            blocks.append(_SourceBlock(text=text, weight=weight, tag=tag.lower()))
        return blocks

    def _extract_text_blocks(self, raw_text: str) -> list[_SourceBlock]:
        blocks: list[_SourceBlock] = []
        for line in raw_text.splitlines():
            text = _normalize_text(line)
            if not text:
                continue
            if text.startswith("#"):
                stripped = text.lstrip("#").strip()
                if stripped:
                    blocks.append(_SourceBlock(text=stripped, weight=0.9, tag="heading"))
                continue
            if text.startswith(("-", "*")):
                stripped = text[1:].strip()
                if stripped:
                    blocks.append(_SourceBlock(text=stripped, weight=0.75, tag="list"))
                continue
            blocks.append(_SourceBlock(text=text, weight=0.5, tag="text"))
        return blocks

    def _extract_terms_from_block(
        self,
        block: _SourceBlock,
        *,
        category: str,
        suffix_hint: bool,
    ) -> list[tuple[str, float]]:
        found: dict[str, float] = {}

        for match in _ORG_TERM_RE.findall(block.text):
            term = match.strip()
            score = min(1.0, block.weight + 0.25)
            found[term] = max(found.get(term, 0.0), score)

        for match in _ASCII_TERM_RE.findall(block.text):
            term = match.strip()
            if term.lower() in _ASCII_STOP_TERMS:
                continue
            score = min(1.0, block.weight + 0.15)
            found[term] = max(found.get(term, 0.0), score)

        for match in _CJK_TERM_RE.findall(block.text):
            term = match.strip()
            if term in _COMMON_STOP_TERMS:
                continue
            score = self._score_cjk_term(term, block=block, category=category, suffix_hint=suffix_hint)
            if score <= 0:
                continue
            found[term] = max(found.get(term, 0.0), score)

        return sorted(found.items(), key=lambda item: (-item[1], -len(item[0]), item[0]))

    def _score_cjk_term(
        self,
        term: str,
        *,
        block: _SourceBlock,
        category: str,
        suffix_hint: bool,
    ) -> float:
        if len(term) < 2 or len(term) > 12:
            return 0.0

        score = block.weight
        if term.endswith(("学院", "大学", "实验室", "研究院", "研究所", "系统", "平台", "课程", "项目")):
            score += 0.25
        elif len(term) in {2, 3, 4} and block.tag in {"title", "h1", "h2", "heading"}:
            score += 0.15

        if category == "person" and 2 <= len(term) <= 4:
            score += 0.12
        if category == "organization" and term.endswith(("学院", "大学", "研究院", "中心")):
            score += 0.12
        if suffix_hint and term.endswith(("系统", "平台", "课程", "项目")):
            score += 0.08

        if term in self.existing_hotwords:
            score -= 0.4
        if len(term) == 2 and block.tag not in {"title", "h1", "h2", "heading"}:
            score -= 0.2
        if term in _COMMON_STOP_TERMS:
            score -= 0.4

        return max(0.0, min(score, 1.0))

    def _merge_discovered(self, discovered: dict[str, list[_CandidateSignal]]) -> int:
        now = datetime.now(timezone.utc).isoformat()
        updated = 0
        for term, signals in discovered.items():
            aggregated = self._aggregate_candidate(term, signals)
            existing = self.candidates.get(term)
            if existing is None:
                self.candidates[term] = aggregated
                updated += 1
                continue

            merged = self._merge_candidate_records(existing, aggregated, now)
            self.candidates[term] = merged
            updated += 1
        return updated

    def _aggregate_candidate(self, term: str, signals: list[_CandidateSignal]) -> dict:
        source_refs = sorted({signal.source_ref for signal in signals})
        contexts = []
        seen_contexts: set[str] = set()
        for signal in signals:
            if signal.context in seen_contexts:
                continue
            seen_contexts.add(signal.context)
            contexts.append(signal.context)
            if len(contexts) >= CANDIDATE_CONTEXT_LIMIT:
                break

        max_signal_score = max(signal.score for signal in signals)
        occurrence_count = len(signals)
        source_count = len(source_refs)
        score = min(1.0, (max_signal_score * 0.55) + (min(occurrence_count, 5) * 0.06) + (min(source_count, 3) * 0.1))
        return {
            "term": term,
            "pinyin": _lazy_pinyin(term),
            "category": signals[0].category,
            "sources": source_refs,
            "count": occurrence_count,
            "contexts": contexts,
            "score": round(score, 4),
            "promoted": False,
            "suggest_promote": score >= AUTO_PROMOTE_THRESHOLD,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }

    def _merge_candidate_records(self, existing: dict, new: dict, now: str) -> dict:
        merged_sources = sorted(set(existing.get("sources", [])) | set(new.get("sources", [])))
        merged_contexts = []
        for context in list(existing.get("contexts", [])) + list(new.get("contexts", [])):
            if context in merged_contexts:
                continue
            merged_contexts.append(context)
            if len(merged_contexts) >= CANDIDATE_CONTEXT_LIMIT:
                break

        merged_count = int(existing.get("count", 0)) + int(new.get("count", 0))
        merged_score = max(float(existing.get("score", 0.0)), float(new.get("score", 0.0)))
        merged_score = min(1.0, merged_score + min(0.15, merged_count * 0.01))

        return {
            "term": new["term"],
            "pinyin": new["pinyin"] or existing.get("pinyin", []),
            "category": new["category"] or existing.get("category", "generic"),
            "sources": merged_sources,
            "count": merged_count,
            "contexts": merged_contexts,
            "score": round(merged_score, 4),
            "promoted": bool(existing.get("promoted", False)),
            "suggest_promote": merged_score >= AUTO_PROMOTE_THRESHOLD,
            "updated_at": now,
        }

    def _write_candidates(self) -> None:
        ordered = sorted(
            self.candidates.values(),
            key=lambda item: (-float(item.get("score", 0.0)), -int(item.get("count", 0)), item.get("term", "")),
        )
        self.candidates_path.write_text(json.dumps(ordered, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Learn hotword candidates from local files and webpages.")
    parser.add_argument("--sources", default=None, help="Optional override for hotword_sources.json")
    parser.add_argument("--candidates", default=None, help="Optional override for hotword_candidates.json")
    parser.add_argument("--hotwords", default=None, help="Optional override for hotwords.json")
    args = parser.parse_args()

    learner = HotwordLearner(
        sources_path=Path(args.sources).resolve() if args.sources else None,
        candidates_path=Path(args.candidates).resolve() if args.candidates else None,
        hotwords_path=Path(args.hotwords).resolve() if args.hotwords else None,
    )
    learner.run()


if __name__ == "__main__":
    main()
