#!/usr/bin/env python3
"""Construct the exact GPT-5.6 Sol Responses Batch requests and cost preflight."""

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
import tiktoken


MODEL = "gpt-5.6-sol"
MAX_OUTPUT_TOKENS = 220
INPUT_PRICE_BATCH_PER_M = 2.00
OUTPUT_PRICE_BATCH_PER_M = 10.00
SUBMISSION_CEILING_USD = 9.00
BUDGET_CEILING_USD = 10.00
EXPECTED_PROMPT_SHA = "f44fbd19f3227569f29f1b0f36d131e164784d91fd81807dd67be1f2e5e13900"
EXPECTED_CLEAN_SHA = "3267f51392dbd02fbfd890d2cb0e5bd514ea9bfdc0c3b79e68a25e76ed674960"
EXPECTED_FROZEN_INPUT_SHA = "8107033004678a66fea15c5889d541fcd2c5fabf205b4913f8ead98f452320ac"
EXPECTED_FROZEN_OUTPUT_SHA = "d1b9bcdce3748b2595e506edb0ec110cbc719ffe8fa0a1e1e20a3a3fe40d481c"
SCORE_KEYS = (
    "correctness_A", "correctness_B",
    "completeness_relevance_A", "completeness_relevance_B",
    "evidence_faithfulness_A", "evidence_faithfulness_B",
)
SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        **{key: {"type": "integer", "minimum": 1, "maximum": 5} for key in SCORE_KEYS},
        "overall_preference": {"type": "string", "enum": ["A", "B", "TIE"]},
        "brief_reason": {"type": "string", "maxLength": 500},
    },
    "required": [*SCORE_KEYS, "overall_preference", "brief_reason"],
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def visible_user_payload(row: dict) -> str:
    return f"""Student question:
{row['question']}

Reference answer:
{row['reference_answer']}

Evidence A:
{row['evidence_A']}

Answer A:
{row['answer_A']}

Evidence B:
{row['evidence_B']}

Answer B:
{row['answer_B']}

Return the JSON evaluation now."""


def request_body(system_prompt: str, row: dict) -> dict:
    return {
        "model": MODEL,
        "instructions": system_prompt,
        "input": visible_user_payload(row),
        "reasoning": {"effort": "none"},
        "max_output_tokens": MAX_OUTPUT_TOKENS,
        "text": {
            "format": {
                "type": "json_schema",
                "name": "pacer_answer_judgment",
                "strict": True,
                "schema": SCHEMA,
            }
        },
        "store": False,
    }


def main() -> None:
    repo_root = Path(__file__).resolve().parents[3]
    root = repo_root / "artifacts" / "automated_judge"
    project = Path(__file__).resolve().parents[3]
    out = root / "openai_gpt56_sol"
    out.mkdir(parents=True, exist_ok=True)
    prompt_path = root / "common_judge_prompt.txt"
    if sha256(prompt_path) != EXPECTED_PROMPT_SHA:
        raise RuntimeError("Common prompt hash differs from the clean Qwen3.6 rubric")
    frozen_dir = project / "outputs/jiis_final_experiments_20260918"
    frozen_input = frozen_dir / "lambdamart_qwen3_generation_inputs_1200.jsonl"
    frozen_output = frozen_dir / "qwen3_lambdamart_generation_1200_final.jsonl"
    if sha256(frozen_input) != EXPECTED_FROZEN_INPUT_SHA or sha256(frozen_output) != EXPECTED_FROZEN_OUTPUT_SHA:
        raise RuntimeError("Frozen generation hash mismatch")

    request_path = out / "batch_requests.jsonl"
    token_path = out / "request_input_token_estimates.csv"
    preflight_path = out / "cost_preflight.json"
    for path in (request_path, token_path, preflight_path):
        if path.exists():
            raise RuntimeError(f"Refusing to overwrite existing preflight artifact: {path}")

    system_prompt = prompt_path.read_text(encoding="utf-8").strip()
    encoding = tiktoken.get_encoding("o200k_base")
    records: list[dict] = []
    token_rows: list[dict] = []
    exact_count_available = bool(os.environ.get("OPENAI_API_KEY"))
    total_offline = 0
    clean_hashes = {}
    for pass_name in ("primary", "swapped"):
        clean_path = root / f"clean_inputs_{pass_name}_600.jsonl"
        clean_hashes[pass_name] = sha256(clean_path)
        rows = read_jsonl(clean_path)
        if len(rows) != 600 or len({row["task_id"] for row in rows}) != 600:
            raise RuntimeError(f"Expected 600 unique {pass_name} rows")
        for row in rows:
            body = request_body(system_prompt, row)
            request = {
                "custom_id": row["judge_item_id"],
                "method": "POST",
                "url": "/v1/responses",
                "body": body,
            }
            # Conservative local count: tokenize the complete canonical body, including
            # field names and JSON syntax. The exact Responses token-count endpoint is
            # preferred when a key is present, but no secret is read or persisted here.
            canonical_body = json.dumps(body, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            offline_count = len(encoding.encode(canonical_body))
            total_offline += offline_count
            records.append(request)
            token_rows.append({
                "custom_id": row["judge_item_id"],
                "pass": pass_name,
                "task_id": row["task_id"],
                "offline_conservative_o200k_base_tokens": offline_count,
                "exact_api_input_tokens": "",
            })

    if len(records) != 1200 or len({record["custom_id"] for record in records}) != 1200:
        raise RuntimeError("Expected exactly 1,200 uniquely identified Batch requests")
    with request_path.open("x", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
    with token_path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(token_rows[0]))
        writer.writeheader()
        writer.writerows(token_rows)

    worst_output_tokens = len(records) * MAX_OUTPUT_TOKENS
    input_cost = total_offline / 1_000_000 * INPUT_PRICE_BATCH_PER_M
    output_cost = worst_output_tokens / 1_000_000 * OUTPUT_PRICE_BATCH_PER_M
    worst_total = input_cost + output_cost
    preflight = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "requested_model": MODEL,
        "api_endpoint": "/v1/responses via /v1/batches",
        "request_count": len(records),
        "primary_requests": 600,
        "swapped_requests": 600,
        "reasoning_effort": "none",
        "max_output_tokens_per_request": MAX_OUTPUT_TOKENS,
        "structured_outputs": True,
        "openai_api_key_present": exact_count_available,
        "token_count_method": "complete canonical Responses body tokenized with tiktoken o200k_base; conservative because JSON field names/syntax are included",
        "token_count_limitation": "The exact /v1/responses/input_tokens endpoint was not called because OPENAI_API_KEY was absent from the current environment.",
        "offline_conservative_input_tokens": total_offline,
        "worst_case_output_tokens": worst_output_tokens,
        "batch_input_price_usd_per_million": INPUT_PRICE_BATCH_PER_M,
        "batch_output_price_usd_per_million": OUTPUT_PRICE_BATCH_PER_M,
        "estimated_batch_input_cost_usd": input_cost,
        "worst_case_batch_output_cost_usd": output_cost,
        "other_known_billable_cost_usd": 0.0,
        "worst_case_total_cost_usd": worst_total,
        "preflight_submission_ceiling_usd": SUBMISSION_CEILING_USD,
        "absolute_budget_ceiling_usd": BUDGET_CEILING_USD,
        "cost_gate_pass_offline_estimate": worst_total <= SUBMISSION_CEILING_USD,
        "submission_authorized": worst_total <= SUBMISSION_CEILING_USD and exact_count_available,
        "submission_blockers": [
            *([] if worst_total <= SUBMISSION_CEILING_USD else ["offline conservative cost exceeds $9.00"]),
            *([] if exact_count_available else ["OPENAI_API_KEY is absent from the current environment"]),
            "exact API input-token counts and model-account availability must be verified before paid submission",
        ],
        "pricing_source": "https://developers.openai.com/api/docs/models/gpt-5.6-sol and OpenAI Batch API 50% discount",
        "pricing_verified_date": "2026-09-19",
        "common_prompt_sha256": sha256(prompt_path),
        "clean_input_sha256": clean_hashes,
        "frozen_generation_input_sha256": EXPECTED_FROZEN_INPUT_SHA,
        "frozen_generation_output_sha256": EXPECTED_FROZEN_OUTPUT_SHA,
        "batch_requests_sha256": sha256(request_path),
        "per_request_token_estimates_sha256": sha256(token_path),
        "python_version": sys.version,
        "platform": platform.platform(),
        "openai_sdk_version": openai.__version__,
        "tiktoken_version": tiktoken.__version__,
    }
    preflight_path.write_text(json.dumps(preflight, indent=2), encoding="utf-8")
    print(json.dumps({
        "requests": len(records),
        "offline_conservative_input_tokens": total_offline,
        "worst_case_output_tokens": worst_output_tokens,
        "worst_case_total_cost_usd": round(worst_total, 6),
        "cost_gate_pass_offline_estimate": preflight["cost_gate_pass_offline_estimate"],
        "openai_api_key_present": exact_count_available,
        "submission_authorized": preflight["submission_authorized"],
    }, indent=2))


if __name__ == "__main__":
    main()
