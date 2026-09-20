#!/usr/bin/env python3
"""Retry the validation-failed 1,200-request job as two sequential 600-item batches."""

from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

from openai import OpenAI


MODEL = "gpt-5.6-sol"
ORIGINAL_FAILED_BATCH = "batch_6aaf26683a488190bdf4fad65c83ae4f"
MAX_ENQUEUED_TOKENS = 1_350_000
TOKENS_PER_PASS = 1_230_931
PREFLIGHT_WORST_CASE_USD = 7.563724
SUBMISSION_CEILING_USD = 9.0


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def save_state(path: Path, state: dict) -> None:
    path.write_text(json.dumps(state, indent=2), encoding="utf-8")


def main() -> None:
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is absent from this shell")
    if PREFLIGHT_WORST_CASE_USD > SUBMISSION_CEILING_USD:
        raise RuntimeError("Preserved total preflight bound exceeds $9")
    if TOKENS_PER_PASS > MAX_ENQUEUED_TOKENS:
        raise RuntimeError("A sequential half still exceeds the account queue limit")

    repo_root = Path(__file__).resolve().parents[3]
    root = repo_root / "artifacts" / "automated_judge"
    out = root / "openai_gpt56_sol"
    manifest = [json.loads(line) for line in (out / "batch_requests.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    groups = {
        pass_name: [row for row in manifest if row["custom_id"].startswith(pass_name + "::")]
        for pass_name in ("primary", "swapped")
    }
    if any(len(rows) != 600 for rows in groups.values()):
        raise RuntimeError("Expected exactly 600 requests in each orientation")
    paths = {name: out / f"batch_requests_{name}_600.jsonl" for name in groups}
    output_paths = {name: out / f"batch_output_{name}_original.jsonl" for name in groups}
    state_path = out / "sequential_batch_execution_metadata.json"
    ids_path = out / "batch_ids_sequential.json"
    if any(path.exists() for path in [*paths.values(), *output_paths.values(), state_path, ids_path]):
        raise RuntimeError("Sequential retry outputs already exist; refusing to overwrite")
    for name, rows in groups.items():
        write_jsonl(paths[name], rows)

    client = OpenAI()
    failed = client.batches.retrieve(ORIGINAL_FAILED_BATCH)
    if failed.status != "failed" or failed.request_counts.completed != 0 or failed.request_counts.failed != 0:
        raise RuntimeError("Original batch state is not the expected zero-request validation failure")
    error_text = str(failed.errors)
    if "token_limit_exceeded" not in error_text:
        raise RuntimeError("Original failure was not the documented enqueued-token limit")

    state = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "requested_model": MODEL,
        "original_batch_id": ORIGINAL_FAILED_BATCH,
        "original_batch_status": failed.status,
        "original_batch_completed_requests": 0,
        "original_batch_failed_requests": 0,
        "original_batch_billable_usage_assumption": 0,
        "original_batch_error": error_text,
        "account_enqueued_token_limit": MAX_ENQUEUED_TOKENS,
        "exact_input_tokens_per_sequential_batch": TOKENS_PER_PASS,
        "combined_exact_input_tokens": TOKENS_PER_PASS * 2,
        "combined_preflight_worst_case_usd": PREFLIGHT_WORST_CASE_USD,
        "submission_ceiling_usd": SUBMISSION_CEILING_USD,
        "execution_order": ["primary", "swapped"],
        "batches": {},
    }
    save_state(state_path, state)

    for pass_name in ("primary", "swapped"):
        with paths[pass_name].open("rb") as handle:
            uploaded = client.files.create(file=handle, purpose="batch")
        batch = client.batches.create(
            input_file_id=uploaded.id,
            endpoint="/v1/responses",
            completion_window="24h",
            metadata={
                "experiment": "PACER_clean_dual_judge_20260919",
                "judge": MODEL,
                "orientation": pass_name,
                "requests": "600",
                "retry_reason": "original_combined_batch_exceeded_account_enqueued_token_limit",
            },
        )
        state["batches"][pass_name] = {
            "batch_id": batch.id, "input_file_id": uploaded.id,
            "submitted_utc": datetime.now(timezone.utc).isoformat(),
            "status_at_submission": batch.status,
        }
        save_state(state_path, state)
        ids_path.write_text(json.dumps({name: data["batch_id"] for name, data in state["batches"].items()}, indent=2), encoding="utf-8")
        while True:
            batch = client.batches.retrieve(batch.id)
            counts = batch.request_counts
            print(
                f"{pass_name.upper()}_BATCH_STATUS={batch.status} "
                f"completed={counts.completed} failed={counts.failed} total={counts.total}",
                flush=True,
            )
            if batch.status in {"completed", "failed", "expired", "cancelled"}:
                break
            time.sleep(30)
        state["batches"][pass_name].update({
            "terminal_status": batch.status,
            "completed_utc": datetime.now(timezone.utc).isoformat(),
            "request_counts": {
                "total": batch.request_counts.total,
                "completed": batch.request_counts.completed,
                "failed": batch.request_counts.failed,
            },
            "output_file_id": batch.output_file_id,
            "error_file_id": batch.error_file_id,
            "errors": str(batch.errors) if batch.errors else None,
        })
        save_state(state_path, state)
        if batch.status != "completed" or batch.request_counts.completed != 600 or batch.request_counts.failed != 0:
            raise RuntimeError(f"{pass_name} sequential batch did not complete cleanly: {batch.status}")
        content = client.files.content(batch.output_file_id).content
        output_paths[pass_name].write_bytes(content)
        lines = [line for line in content.decode("utf-8").splitlines() if line.strip()]
        if len(lines) != 600:
            raise RuntimeError(f"{pass_name} output file contains {len(lines)} lines, expected 600")
        if batch.error_file_id:
            (out / f"batch_error_{pass_name}_original.jsonl").write_bytes(client.files.content(batch.error_file_id).content)
        print(f"{pass_name.upper()}_BATCH_COLLECTION=PASS batch_id={batch.id} lines=600", flush=True)

    state["completed_utc"] = datetime.now(timezone.utc).isoformat()
    state["status"] = "completed"
    save_state(state_path, state)
    print("OPENAI_SEQUENTIAL_BATCHES = PASS total=1200")


if __name__ == "__main__":
    main()
