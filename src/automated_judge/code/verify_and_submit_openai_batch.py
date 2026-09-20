#!/usr/bin/env python3
"""Verify GPT-5.6 Sol access, exact-count inputs, enforce $9 gate, then submit Batch."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import openai
from openai import OpenAI


MODEL = "gpt-5.6-sol"
INPUT_PRICE_BATCH_PER_M = 2.00
OUTPUT_PRICE_BATCH_PER_M = 10.00
MAX_OUTPUT_TOKENS = 220
REQUESTS = 1200
SUBMISSION_CEILING_USD = 9.00
BUDGET_CEILING_USD = 10.00


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is absent; no API call or submission was made")
    repo_root = Path(__file__).resolve().parents[3]
    root = repo_root / "artifacts" / "automated_judge"
    out = root / "openai_gpt56_sol"
    request_path = out / "batch_requests.jsonl"
    offline_path = out / "cost_preflight_offline.json"
    exact_path = out / "cost_preflight.json"
    exact_counts_path = out / "request_input_token_counts_exact.csv"
    batch_id_path = out / "batch_id.txt"
    submission_path = out / "batch_submission_metadata.json"
    for path in (exact_path, exact_counts_path, batch_id_path, submission_path):
        if path.exists():
            raise RuntimeError(f"Refusing to overwrite existing API artifact: {path}")
    if not offline_path.is_file() or not request_path.is_file():
        raise RuntimeError("Offline preflight artifacts are missing")

    client = OpenAI()
    # Account-specific availability check. No alternate model is attempted.
    try:
        model_record = client.models.retrieve(MODEL)
    except Exception as exc:
        raise RuntimeError(f"Requested model {MODEL!r} is unavailable to this API account: {type(exc).__name__}: {exc}") from exc

    requests = [json.loads(line) for line in request_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(requests) != REQUESTS or len({row["custom_id"] for row in requests}) != REQUESTS:
        raise RuntimeError("Batch request manifest is not exactly 1,200 unique requests")

    total_input = 0
    count_rows = []
    for index, request in enumerate(requests, start=1):
        body = request["body"]
        count = client.responses.input_tokens.count(
            model=body["model"],
            instructions=body["instructions"],
            input=body["input"],
            reasoning=body["reasoning"],
            text=body["text"],
        )
        tokens = int(count.input_tokens)
        total_input += tokens
        pass_name, task_id = request["custom_id"].split("::", 1)
        count_rows.append({
            "custom_id": request["custom_id"],
            "pass": pass_name,
            "task_id": task_id,
            "exact_api_input_tokens": tokens,
        })
        if index % 100 == 0:
            print(f"EXACT_TOKEN_COUNT_PROGRESS={index}/{REQUESTS}", flush=True)
    with exact_counts_path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(count_rows[0]))
        writer.writeheader()
        writer.writerows(count_rows)

    worst_output = REQUESTS * MAX_OUTPUT_TOKENS
    input_cost = total_input / 1_000_000 * INPUT_PRICE_BATCH_PER_M
    output_cost = worst_output / 1_000_000 * OUTPUT_PRICE_BATCH_PER_M
    worst_total = input_cost + output_cost
    preflight = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "requested_model": MODEL,
        "account_model_retrieve_id": getattr(model_record, "id", None),
        "model_availability_verified": True,
        "batch_support_documented": True,
        "api_endpoint": "/v1/responses via /v1/batches",
        "request_count": REQUESTS,
        "primary_requests": 600,
        "swapped_requests": 600,
        "reasoning_effort": "none",
        "max_output_tokens_per_request": MAX_OUTPUT_TOKENS,
        "exact_api_input_tokens": total_input,
        "worst_case_output_tokens": worst_output,
        "batch_input_price_usd_per_million": INPUT_PRICE_BATCH_PER_M,
        "batch_output_price_usd_per_million": OUTPUT_PRICE_BATCH_PER_M,
        "estimated_batch_input_cost_usd": input_cost,
        "worst_case_batch_output_cost_usd": output_cost,
        "other_known_billable_cost_usd": 0.0,
        "worst_case_total_cost_usd": worst_total,
        "preflight_submission_ceiling_usd": SUBMISSION_CEILING_USD,
        "absolute_budget_ceiling_usd": BUDGET_CEILING_USD,
        "cost_gate_pass": worst_total <= SUBMISSION_CEILING_USD,
        "pricing_source": "https://developers.openai.com/api/docs/models/gpt-5.6-sol and OpenAI Batch API 50% discount",
        "pricing_verified_date": "2026-09-19",
        "batch_requests_sha256": sha256(request_path),
        "exact_token_counts_sha256": sha256(exact_counts_path),
        "openai_sdk_version": openai.__version__,
        "python_version": sys.version,
        "platform": platform.platform(),
    }
    exact_path.write_text(json.dumps(preflight, indent=2), encoding="utf-8")
    if worst_total > SUBMISSION_CEILING_USD:
        print("OPENAI_SOL_COST_GATE_ABORT = TRUE")
        print(f"worst_case_total_cost_usd={worst_total:.6f}")
        return

    with request_path.open("rb") as handle:
        input_file = client.files.create(file=handle, purpose="batch")
    batch = client.batches.create(
        input_file_id=input_file.id,
        endpoint="/v1/responses",
        completion_window="24h",
        metadata={
            "experiment": "PACER_clean_dual_judge_20260919",
            "judge": "gpt-5.6-sol",
            "requests": "1200",
        },
    )
    batch_id_path.write_text(batch.id + "\n", encoding="utf-8")
    submission = {
        "submitted_utc": datetime.now(timezone.utc).isoformat(),
        "requested_model": MODEL,
        "account_model_retrieve_id": getattr(model_record, "id", None),
        "openai_sdk_version": openai.__version__,
        "endpoint": "/v1/responses via /v1/batches",
        "input_file_id": input_file.id,
        "batch_id": batch.id,
        "batch_status_at_submission": batch.status,
        "completion_window": batch.completion_window,
        "request_count": REQUESTS,
        "exact_input_tokens": total_input,
        "preflight_worst_case_cost_usd": worst_total,
        "reasoning_effort": "none",
        "max_output_tokens": MAX_OUTPUT_TOKENS,
        "api_key_stored": False,
    }
    submission_path.write_text(json.dumps(submission, indent=2), encoding="utf-8")
    print(f"OPENAI_BATCH_SUBMITTED={batch.id}")
    print(f"PREFLIGHT_WORST_CASE_USD={worst_total:.6f}")


if __name__ == "__main__":
    main()
