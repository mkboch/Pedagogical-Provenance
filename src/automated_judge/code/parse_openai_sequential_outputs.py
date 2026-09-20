#!/usr/bin/env python3
"""Parse, validate, and cost the two successful sequential GPT-5.6 Sol batches."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


INPUT_BATCH_RATE = 2.0
CACHED_INPUT_BATCH_RATE = 0.2
OUTPUT_BATCH_RATE = 10.0
SCORE_KEYS = (
    "correctness_A", "correctness_B",
    "completeness_relevance_A", "completeness_relevance_B",
    "evidence_faithfulness_A", "evidence_faithfulness_B",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def extract_text(body: dict) -> str:
    texts = []
    for item in body.get("output", []):
        if item.get("type") == "message":
            for content in item.get("content", []):
                if content.get("type") in {"output_text", "text"} and isinstance(content.get("text"), str):
                    texts.append(content["text"])
    if not texts and isinstance(body.get("output_text"), str):
        texts.append(body["output_text"])
    return "".join(texts)


def validate(text: str) -> tuple[dict | None, str | None]:
    try:
        parsed = json.loads(text)
    except Exception as exc:
        return None, f"json_parse:{type(exc).__name__}:{exc}"
    required = {*SCORE_KEYS, "overall_preference", "brief_reason"}
    if not isinstance(parsed, dict) or set(parsed) != required:
        return None, "field_set"
    for key in SCORE_KEYS:
        if isinstance(parsed[key], bool) or not isinstance(parsed[key], int) or not 1 <= parsed[key] <= 5:
            return None, f"invalid_score:{key}"
    if parsed["overall_preference"] not in {"A", "B", "TIE"}:
        return None, "invalid_preference"
    if not isinstance(parsed["brief_reason"], str):
        return None, "invalid_reason_type"
    if len(parsed["brief_reason"].split()) > 50:
        return None, "reason_over_50_words"
    return parsed, None


def main() -> None:
    repo_root = Path(__file__).resolve().parents[3]
    root = repo_root / "artifacts" / "automated_judge"
    out = root / "openai_gpt56_sol"
    execution = json.loads((out / "sequential_batch_execution_metadata.json").read_text(encoding="utf-8"))
    if execution.get("status") != "completed":
        raise RuntimeError("Sequential Batch execution is not complete")
    raw_targets = {name: out / f"openai_{name}_raw_600.jsonl" for name in ("primary", "swapped")}
    metadata_path = out / "openai_collection_metadata.json"
    failure_path = out / "failed_requests_for_retry.json"
    actual_cost_path = out / "ACTUAL_COST_REPORT.json"
    openai_metadata_path = out / "openai_metadata.json"
    for path in [*raw_targets.values(), metadata_path, failure_path, actual_cost_path, openai_metadata_path]:
        if path.exists():
            raise RuntimeError(f"Refusing to overwrite {path}")

    totals = {
        "input_tokens": 0, "cached_input_tokens": 0, "output_tokens": 0,
        "reasoning_tokens": 0, "total_tokens": 0,
    }
    returned_models = set()
    failures = []
    all_records = {}
    output_hashes = {}
    for pass_name in ("primary", "swapped"):
        source = out / f"batch_output_{pass_name}_original.jsonl"
        output_hashes[pass_name] = sha256(source)
        lines = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines() if line.strip()]
        if len(lines) != 600 or len({line["custom_id"] for line in lines}) != 600:
            raise RuntimeError(f"Expected 600 unique {pass_name} Batch outputs")
        records = []
        for line in lines:
            embedded_pass, task_id = line["custom_id"].split("::", 1)
            if embedded_pass != pass_name:
                raise RuntimeError("Batch output orientation mismatch")
            response = line.get("response")
            api_error = line.get("error")
            body = response.get("body") if response else None
            raw_text = ""
            parsed = None
            validation_error = None
            if api_error is not None or not response or response.get("status_code") != 200 or not body:
                validation_error = f"api_error:{api_error or response}"
            else:
                raw_text = extract_text(body)
                parsed, validation_error = validate(raw_text)
                if body.get("model"):
                    returned_models.add(body["model"])
                usage = body.get("usage") or {}
                in_details = usage.get("input_tokens_details") or {}
                out_details = usage.get("output_tokens_details") or {}
                totals["input_tokens"] += int(usage.get("input_tokens") or 0)
                totals["cached_input_tokens"] += int(in_details.get("cached_tokens") or 0)
                totals["output_tokens"] += int(usage.get("output_tokens") or 0)
                totals["reasoning_tokens"] += int(out_details.get("reasoning_tokens") or 0)
                totals["total_tokens"] += int(usage.get("total_tokens") or 0)
            record = {
                "judge_item_id": line["custom_id"], "pass": pass_name, "task_id": task_id,
                "batch_request_id": line.get("id"), "request_id": response.get("request_id") if response else None,
                "status_code": response.get("status_code") if response else None,
                "attempts": [{
                    "attempt": 1, "raw_output": raw_text, "validation_error": validation_error,
                    "api_error": api_error, "response_body": body,
                }],
                "selected_attempt": 1 if validation_error is None else None,
                "parsed": parsed,
            }
            records.append(record)
            if validation_error is not None:
                failures.append({"custom_id": line["custom_id"], "validation_error": validation_error})
        records.sort(key=lambda record: record["task_id"])
        all_records[pass_name] = records
        with raw_targets[pass_name].open("x", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    failure_path.write_text(json.dumps(failures, indent=2), encoding="utf-8")

    uncached = totals["input_tokens"] - totals["cached_input_tokens"]
    conservative_cost = totals["input_tokens"] / 1_000_000 * INPUT_BATCH_RATE + totals["output_tokens"] / 1_000_000 * OUTPUT_BATCH_RATE
    cache_adjusted_cost = (
        uncached / 1_000_000 * INPUT_BATCH_RATE
        + totals["cached_input_tokens"] / 1_000_000 * CACHED_INPUT_BATCH_RATE
        + totals["output_tokens"] / 1_000_000 * OUTPUT_BATCH_RATE
    )
    batch_ids = {name: execution["batches"][name]["batch_id"] for name in ("primary", "swapped")}
    metadata = {
        "collected_utc": datetime.now(timezone.utc).isoformat(),
        "requested_model": "gpt-5.6-sol",
        "returned_model_identifiers": sorted(returned_models),
        "batch_ids": batch_ids,
        "original_validation_failed_batch_id": execution["original_batch_id"],
        "request_counts": {"primary": 600, "swapped": 600, "successful": 1200, "failed": 0},
        "usage": totals,
        "parser_failure_count": len(failures),
        "conservative_estimated_actual_cost_usd": conservative_cost,
        "cache_adjusted_estimated_actual_cost_usd": cache_adjusted_cost,
        "batch_output_sha256": output_hashes,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    actual_cost = {
        "model": "gpt-5.6-sol", "returned_model_identifiers": sorted(returned_models),
        "budget_ceiling_usd": 10.0, "preflight_submission_ceiling_usd": 9.0,
        "preflight_worst_case_estimate_usd": 7.563724,
        "actual_usage": totals,
        "batch_rates_usd_per_million": {
            "input_uncached": INPUT_BATCH_RATE,
            "input_cached": CACHED_INPUT_BATCH_RATE,
            "output_including_reasoning": OUTPUT_BATCH_RATE,
        },
        "estimated_actual_charge_conservative_usd": conservative_cost,
        "estimated_actual_charge_cache_adjusted_usd": cache_adjusted_cost,
        "reasoning_cost_note": "Reasoning tokens are included in output_tokens and are not added a second time.",
        "pricing_source": "https://developers.openai.com/api/docs/models/gpt-5.6-sol and OpenAI Batch API 50% discount",
        "pricing_verified_date": "2026-09-19",
        "within_nine_dollar_ceiling": conservative_cost <= 9.0,
    }
    actual_cost_path.write_text(json.dumps(actual_cost, indent=2), encoding="utf-8")
    openai_metadata = {
        "status": "complete" if not failures else "parser_gate_failed",
        "requested_model": "gpt-5.6-sol",
        "returned_model_identifiers": sorted(returned_models),
        "api_endpoint": "/v1/responses via /v1/batches",
        "batch_ids": batch_ids,
        "original_validation_failed_batch_id": execution["original_batch_id"],
        "reasoning_configuration": {"effort": "none"},
        "max_output_tokens": 220,
        "request_counts": metadata["request_counts"],
        "usage": totals,
        "estimated_actual_charge_conservative_usd": conservative_cost,
        "preflight_worst_case_cost_usd": 7.563724,
        "parser_failure_count": len(failures),
        "api_key_stored": False,
    }
    openai_metadata_path.write_text(json.dumps(openai_metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))
    if failures:
        raise RuntimeError(f"OpenAI parser gate failed for {len(failures)} judgments")
    print("OPENAI_PARSE_GATE = PASS valid=1200")


if __name__ == "__main__":
    main()
