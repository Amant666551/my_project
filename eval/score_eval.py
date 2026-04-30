from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path


def _load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as fp:
        for line_no, raw in enumerate(fp, start=1):
            line = raw.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_no}: {exc}") from exc
    return rows


def _normalize_label(value: object, default: str) -> str:
    text = str(value or "").strip()
    return text if text else default


def _normalize_route_label(value: object, default: str) -> str:
    text = _normalize_label(value, default)
    lowered = text.lower()
    if lowered.startswith("voice_"):
        suffix = text.split("_", 1)[1].strip()
        return suffix or default
    return text


def _tokenize(text: str) -> list[str]:
    pattern = re.compile(r"[\u4e00-\u9fff]|[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?")
    return pattern.findall((text or "").lower())


def _token_f1(reference: str, prediction: str) -> float:
    ref_tokens = _tokenize(reference)
    pred_tokens = _tokenize(prediction)
    if not ref_tokens and not pred_tokens:
        return 1.0
    if not ref_tokens or not pred_tokens:
        return 0.0

    ref_counter = Counter(ref_tokens)
    pred_counter = Counter(pred_tokens)
    overlap = sum((ref_counter & pred_counter).values())
    precision = overlap / max(1, len(pred_tokens))
    recall = overlap / max(1, len(ref_tokens))
    if precision + recall == 0.0:
        return 0.0
    return (2.0 * precision * recall) / (precision + recall)


def _corpus_bleu(references: list[str], predictions: list[str], max_order: int = 4) -> float:
    ref_corpus = [_tokenize(text) for text in references]
    pred_corpus = [_tokenize(text) for text in predictions]
    if not ref_corpus or not pred_corpus:
        return 0.0

    matches_by_order = [0] * max_order
    possible_matches_by_order = [0] * max_order
    ref_length = 0
    pred_length = 0

    for ref_tokens, pred_tokens in zip(ref_corpus, pred_corpus):
        ref_length += len(ref_tokens)
        pred_length += len(pred_tokens)
        for order in range(1, max_order + 1):
            ref_ngrams = Counter(
                tuple(ref_tokens[i : i + order]) for i in range(max(0, len(ref_tokens) - order + 1))
            )
            pred_ngrams = Counter(
                tuple(pred_tokens[i : i + order]) for i in range(max(0, len(pred_tokens) - order + 1))
            )
            overlap = pred_ngrams & ref_ngrams
            matches_by_order[order - 1] += sum(overlap.values())
            possible_matches_by_order[order - 1] += max(0, len(pred_tokens) - order + 1)

    precisions: list[float] = []
    for matches, possible in zip(matches_by_order, possible_matches_by_order):
        if possible == 0:
            precisions.append(0.0)
        elif matches == 0:
            precisions.append(1e-9)
        else:
            precisions.append(matches / possible)

    if min(precisions) <= 0.0:
        geo_mean = 0.0
    else:
        geo_mean = math.exp(sum(math.log(p) for p in precisions) / max_order)

    if pred_length == 0:
        return 0.0
    if pred_length > ref_length:
        bp = 1.0
    else:
        bp = math.exp(1.0 - (ref_length / max(1, pred_length)))

    return 100.0 * bp * geo_mean


def _percent(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return 100.0 * numerator / denominator


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _percentile(values: list[float], ratio: float) -> float:
    if not values:
        return 0.0
    sorted_values = sorted(values)
    index = min(len(sorted_values) - 1, max(0, math.ceil(ratio * len(sorted_values)) - 1))
    return sorted_values[index]


def _align_rows(ref_rows: list[dict], pred_rows: list[dict]) -> list[tuple[dict, dict]]:
    def has_key(rows: list[dict], key: str) -> bool:
        return bool(rows) and all(key in row for row in rows)

    for key in ("utt_id", "utt_index"):
        if has_key(ref_rows, key) and has_key(pred_rows, key):
            pred_map = {str(row[key]): row for row in pred_rows}
            pairs = []
            for ref in ref_rows:
                match = pred_map.get(str(ref[key]))
                if match is not None:
                    pairs.append((ref, match))
            return pairs

    return list(zip(ref_rows, pred_rows))


def evaluate(ref_rows: list[dict], pred_rows: list[dict]) -> dict:
    pairs = _align_rows(ref_rows, pred_rows)
    total = len(pairs)
    if total == 0:
        raise ValueError("No aligned utterances found between refs and preds.")

    route_hits = 0
    unknown_count = 0
    exact_match_hits = 0
    token_f1_scores: list[float] = []
    bleu_refs: list[str] = []
    bleu_preds: list[str] = []
    end_to_end_latencies: list[float] = []
    speaker_groups: dict[str, list[str]] = defaultdict(list)

    ref_switches = 0
    pred_switches = 0
    true_positive_switches = 0
    false_positive_switches = 0
    no_switch_positions = 0

    previous_ref_route: str | None = None
    previous_pred_route: str | None = None

    for ref, pred in pairs:
        route_ref = _normalize_route_label(ref.get("route_ref"), "none")
        speaker_ref = _normalize_label(ref.get("speaker_ref"), "unknown_ref")
        en_ref = str(ref.get("en_ref", "")).strip()

        route_pred = _normalize_route_label(pred.get("route_pred"), "none")
        mt_pred = str(pred.get("translated_text", pred.get("mt", ""))).strip()

        if route_pred == route_ref:
            route_hits += 1
        if route_pred == "none" or _normalize_label(pred.get("speaker_pred"), "unknown") == "unknown":
            unknown_count += 1
        if mt_pred == en_ref and en_ref:
            exact_match_hits += 1

        token_f1_scores.append(_token_f1(en_ref, mt_pred))
        bleu_refs.append(en_ref)
        bleu_preds.append(mt_pred)
        speaker_groups[speaker_ref].append(route_pred)

        if "end_to_end_latency_ms" in pred and pred.get("end_to_end_latency_ms") is not None:
            try:
                end_to_end_latencies.append(float(pred["end_to_end_latency_ms"]))
            except (TypeError, ValueError):
                pass

        if previous_ref_route is not None and previous_pred_route is not None:
            ref_switched = route_ref != previous_ref_route
            pred_switched = route_pred != previous_pred_route
            if ref_switched:
                ref_switches += 1
            else:
                no_switch_positions += 1
            if pred_switched:
                pred_switches += 1
            if ref_switched and pred_switched:
                true_positive_switches += 1
            if (not ref_switched) and pred_switched:
                false_positive_switches += 1

        previous_ref_route = route_ref
        previous_pred_route = route_pred

    consistency_scores = []
    for routes in speaker_groups.values():
        if not routes:
            continue
        modal_count = Counter(routes).most_common(1)[0][1]
        consistency_scores.append(modal_count / len(routes))

    switch_precision = true_positive_switches / pred_switches if pred_switches else 0.0
    switch_recall = true_positive_switches / ref_switches if ref_switches else 0.0
    if switch_precision + switch_recall == 0.0:
        switch_f1 = 0.0
    else:
        switch_f1 = 2.0 * switch_precision * switch_recall / (switch_precision + switch_recall)

    return {
        "utterances_scored": total,
        "route_accuracy": round(_percent(route_hits, total), 3),
        "unknown_rate": round(_percent(unknown_count, total), 3),
        "speaker_consistency": round(100.0 * _mean(consistency_scores), 3),
        "false_switch_rate": round(_percent(false_positive_switches, no_switch_positions), 3),
        "switch_precision": round(100.0 * switch_precision, 3),
        "switch_recall": round(100.0 * switch_recall, 3),
        "switch_f1": round(100.0 * switch_f1, 3),
        "translation_exact_match": round(_percent(exact_match_hits, total), 3),
        "translation_token_f1": round(100.0 * _mean(token_f1_scores), 3),
        "translation_bleu": round(_corpus_bleu(bleu_refs, bleu_preds), 3),
        "end_to_end_latency_ms_mean": round(_mean(end_to_end_latencies), 3),
        "end_to_end_latency_ms_p95": round(_percentile(end_to_end_latencies, 0.95), 3),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Score realtime speech translation evaluation runs.")
    parser.add_argument("--refs", required=True, help="Reference JSONL file.")
    parser.add_argument("--preds", required=True, help="Prediction JSONL file.")
    parser.add_argument("--out", help="Optional output JSON file.")
    args = parser.parse_args()

    refs_path = Path(args.refs)
    preds_path = Path(args.preds)
    ref_rows = _load_jsonl(refs_path)
    pred_rows = _load_jsonl(preds_path)
    metrics = evaluate(ref_rows, pred_rows)

    print(f"refs_file={refs_path}")
    print(f"preds_file={preds_path}")
    for key, value in metrics.items():
        print(f"{key}={value}")

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "refs_file": str(refs_path),
            "preds_file": str(preds_path),
            "metrics": metrics,
        }
        out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"metrics_json={out_path}")


if __name__ == "__main__":
    main()
