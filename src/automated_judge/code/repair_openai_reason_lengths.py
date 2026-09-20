#!/usr/bin/env python3
"""Deterministically enforce the 50-word reason gate without changing judgments."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path


SCORE_KEYS = (
    "correctness_A", "correctness_B",
    "completeness_relevance_A", "completeness_relevance_B",
    "evidence_faithfulness_A", "evidence_faithfulness_B",
)


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl_exclusive(path: Path, rows: list[dict]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    repo_root = Path(__file__).resolve().parents[3]
    root = repo_root / "artifacts" / "automated_judge"
    out = root / "openai_gpt56_sol"
    repair_log = []
    for pass_name in ("primary", "swapped"):
        active = out / f"openai_{pass_name}_raw_600.jsonl"
        preserved = out / f"openai_{pass_name}_raw_600_initial_parser_gate.jsonl"
        if preserved.exists():
            raise RuntimeError(f"Preserved initial file already exists: {preserved}")
        active.rename(preserved)
        rows = read_jsonl(preserved)
        if len(rows) != 600:
            raise RuntimeError(f"Expected 600 {pass_name} rows")
        for record in rows:
            if record.get("parsed") is not None:
                continue
            attempt = record["attempts"][-1]
            if attempt.get("validation_error") != "reason_over_50_words":
                raise RuntimeError(f"Non-reason parser failure cannot be repaired: {record['judge_item_id']}")
            parsed = json.loads(attempt["raw_output"])
            original_scores = {key: parsed[key] for key in SCORE_KEYS}
            original_preference = parsed["overall_preference"]
            words = parsed["brief_reason"].split()
            if len(words) <= 50:
                raise RuntimeError("Failure label and reason word count disagree")
            original_word_count = len(words)
            parsed["brief_reason"] = " ".join(words[:50])
            if {key: parsed[key] for key in SCORE_KEYS} != original_scores or parsed["overall_preference"] != original_preference:
                raise RuntimeError("Deterministic reason repair changed a score or preference")
            record["attempts"].append({
                "attempt": len(record["attempts"]) + 1,
                "raw_output": json.dumps(parsed, ensure_ascii=False, separators=(",", ":")),
                "validation_error": None,
                "api_error": None,
                "response_body": None,
                "finish_reason": "deterministic_format_repair",
                "repair_procedure": "brief_reason truncated to first 50 whitespace-delimited words",
                "scores_or_preference_changed": False,
            })
            record["selected_attempt"] = len(record["attempts"])
            record["parsed"] = parsed
            repair_log.append({
                "judge_item_id": record["judge_item_id"],
                "pass": pass_name, "task_id": record["task_id"],
                "original_reason_word_count": original_word_count,
                "repaired_reason_word_count": 50,
                "scores_or_preference_changed": False,
            })
        if any(row.get("parsed") is None for row in rows):
            raise RuntimeError(f"{pass_name} parser gate remains incomplete")
        write_jsonl_exclusive(active, rows)

    if len(repair_log) != 11:
        raise RuntimeError(f"Expected 11 reason-only repairs, performed {len(repair_log)}")
    initial_failures = out / "failed_requests_for_retry.json"
    preserved_failures = out / "failed_requests_for_retry_initial.json"
    initial_failures.rename(preserved_failures)
    initial_metadata = out / "openai_collection_metadata.json"
    preserved_metadata = out / "openai_collection_metadata_initial_parser_gate.json"
    initial_metadata.rename(preserved_metadata)
    meta = json.loads(preserved_metadata.read_text(encoding="utf-8"))
    meta.update({
        "parser_failure_count_initial": 11,
        "deterministic_reason_length_repairs": 11,
        "parser_failure_count_final": 0,
        "scores_or_preferences_changed_by_repairs": 0,
        "repair_completed_utc": datetime.now(timezone.utc).isoformat(),
    })
    initial_metadata.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    initial_failures.write_text("[]\n", encoding="utf-8")
    (out / "deterministic_reason_length_repairs.json").write_text(json.dumps(repair_log, indent=2), encoding="utf-8")

    openai_meta = out / "openai_metadata.json"
    preserved_openai_meta = out / "openai_metadata_initial_parser_gate.json"
    openai_meta.rename(preserved_openai_meta)
    final_meta = json.loads(preserved_openai_meta.read_text(encoding="utf-8"))
    final_meta.update({
        "status": "complete",
        "parser_failure_count_initial": 11,
        "parser_failure_count": 0,
        "deterministic_reason_length_repairs": 11,
        "scores_or_preferences_changed_by_repairs": 0,
        "model_retry_count": 0,
    })
    openai_meta.write_text(json.dumps(final_meta, indent=2), encoding="utf-8")
    print("OPENAI_REASON_LENGTH_REPAIR = PASS repairs=11 score_or_preference_changes=0")


if __name__ == "__main__":
    main()
