#!/usr/bin/env python3
"""Poll and collect the submitted GPT-5.6 Sol Batch without exposing credentials."""

from __future__ import annotations

import hashlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import openai
from openai import OpenAI


EXPECTED_BATCH_ID = "batch_6aaf26683a488190bdf4fad65c83ae4f"
EXPECTED_REQUESTS = 1200
INPUT_BATCH_RATE = 2.0
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
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is absent from this shell")
    repo_root = Path(__file__).resolve().parents[3]
    root = repo_root / "artifacts" / "automated_judge"
    out = root / "openai_gpt56_sol"
    batch_id = (out / "batch_id.txt").read_text(encoding="utf-8").strip()
    if batch_id != EXPECTED_BATCH_ID:
        raise RuntimeError(f"Unexpected batch ID: {batch_id}")
    targets = [
        out / "batch_output_original.jsonl", out / "batch_error_original.jsonl",
        out / "openai_primary_raw_600.jsonl", out / "openai_swapped_raw_600.jsonl",
        out / "openai_collection_metadata.json", out / "failed_requests_for_retry.json",
    ]
    if any(path.exists() for path in targets):
        raise RuntimeError("One or more collection outputs already exist; refusing to overwrite")

    client = OpenAI()
    while True:
        batch = client.batches.retrieve(batch_id)
        counts = batch.request_counts
        print(
            f"BATCH_STATUS={batch.status} completed={getattr(counts, 'completed', None)} "
            f"failed={getattr(counts, 'failed', None)} total={getattr(counts, 'total', None)}",
            flush=True,
        )
        if batch.status in {"completed", "failed", "expired", "cancelled"}:
            break
        time.sleep(30)
    if batch.status != "completed":
        raise RuntimeError(f"Batch ended in terminal status {batch.status}: {batch.errors}")
    if not batch.output_file_id:
        raise RuntimeError("Completed batch has no output file")

    output_bytes = client.files.content(batch.output_file_id).content
    original_output = out / "batch_output_original.jsonl"
    original_output.write_bytes(output_bytes)
    if batch.error_file_id:
        (out / "batch_error_original.jsonl").write_bytes(client.files.content(batch.error_file_id).content)

    lines = [json.loads(line) for line in output_bytes.decode("utf-8").splitlines() if line.strip()]
    if len(lines) != EXPECTED_REQUESTS or len({line["custom_id"] for line in lines}) != EXPECTED_REQUESTS:
        raise RuntimeError(f"Expected 1,200 unique returned lines, found {len(lines)}")
    records = {"primary": [], "swapped": []}
    failures = []
    usage_totals = {
        "input_tokens": 0, "cached_input_tokens": 0, "output_tokens": 0,
        "reasoning_tokens": 0, "total_tokens": 0,
    }
    returned_models = set()
    for line in lines:
        pass_name, task_id = line["custom_id"].split("::", 1)
        response = line.get("response")
        api_error = line.get("error")
        parsed = None
        validation_error = None
        raw_text = ""
        body = None
        if api_error is not None or not response or response.get("status_code") != 200:
            validation_error = f"api_error:{api_error or response}"
        else:
            body = response["body"]
            returned_models.add(body.get("model"))
            raw_text = extract_text(body)
            parsed, validation_error = validate(raw_text)
            usage = body.get("usage") or {}
            input_details = usage.get("input_tokens_details") or {}
            output_details = usage.get("output_tokens_details") or {}
            usage_totals["input_tokens"] += int(usage.get("input_tokens") or 0)
            usage_totals["cached_input_tokens"] += int(input_details.get("cached_tokens") or 0)
            usage_totals["output_tokens"] += int(usage.get("output_tokens") or 0)
            usage_totals["reasoning_tokens"] += int(output_details.get("reasoning_tokens") or 0)
            usage_totals["total_tokens"] += int(usage.get("total_tokens") or 0)
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
        records[pass_name].append(record)
        if validation_error is not None:
            failures.append({"custom_id": line["custom_id"], "validation_error": validation_error})

    for pass_name in ("primary", "swapped"):
        rows = sorted(records[pass_name], key=lambda row: row["task_id"])
        if len(rows) != 600:
            raise RuntimeError(f"Expected 600 {pass_name} records, found {len(rows)}")
        with (out / f"openai_{pass_name}_raw_600.jsonl").open("x", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    (out / "failed_requests_for_retry.json").write_text(json.dumps(failures, indent=2), encoding="utf-8")
    conservative_actual_cost = (
        usage_totals["input_tokens"] / 1_000_000 * INPUT_BATCH_RATE
        + usage_totals["output_tokens"] / 1_000_000 * OUTPUT_BATCH_RATE
    )
    metadata = {
        "collected_utc": datetime.now(timezone.utc).isoformat(),
        "batch_id": batch_id, "batch_status": batch.status,
        "request_counts": {
            "total": batch.request_counts.total,
            "completed": batch.request_counts.completed,
            "failed": batch.request_counts.failed,
        },
        "returned_model_identifiers": sorted(model for model in returned_models if model),
        "usage": usage_totals,
        "conservative_estimated_actual_cost_usd": conservative_actual_cost,
        "parser_or_api_failure_count": len(failures),
        "batch_output_sha256": sha256(original_output),
        "openai_sdk_version": openai.__version__,
    }
    (out / "openai_collection_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2), flush=True)
    if failures:
        raise RuntimeError(f"Collection preserved all outputs but {len(failures)} requests require retry")
    print("OPENAI_COLLECTION = PASS valid=1200")


if __name__ == "__main__":
    main()
