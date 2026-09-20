#!/usr/bin/env python3
"""Build one clean, condition-blinded paired dataset from frozen PACER artifacts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path


SEED = 20260919
EXPECTED_INPUT_SHA = "8107033004678a66fea15c5889d541fcd2c5fabf205b4913f8ead98f452320ac"
EXPECTED_OUTPUT_SHA = "d1b9bcdce3748b2595e506edb0ec110cbc719ffe8fa0a1e1e20a3a3fe40d481c"
CONDITIONS = ("content_only_lambdamart", "full_pacer_lambdamart")
SUBSTANTIVE_FIELDS = ("question", "reference_answer", "evidence_A", "answer_A", "evidence_B", "answer_B")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")



def main() -> None:
    parser = argparse.ArgumentParser()
    repo_root = Path(__file__).resolve().parents[3]
    parser.add_argument("--project-root", type=Path, default=repo_root)
    parser.add_argument("--output-root", type=Path, default=repo_root / "artifacts" / "automated_judge")
    args = parser.parse_args()
    project = args.project_root.resolve()
    out = args.output_root.resolve()
    out.mkdir(parents=True, exist_ok=True)

    frozen_dir = project / "outputs/jiis_final_experiments_20260918"
    input_path = frozen_dir / "lambdamart_qwen3_generation_inputs_1200.jsonl"
    output_path = frozen_dir / "qwen3_lambdamart_generation_1200_final.jsonl"
    if sha256(input_path) != EXPECTED_INPUT_SHA or sha256(output_path) != EXPECTED_OUTPUT_SHA:
        raise RuntimeError("Frozen generation hash mismatch; stopping before construction")

    inputs = read_jsonl(input_path)
    outputs = read_jsonl(output_path)
    if len(inputs) != 1200 or len(outputs) != 1200:
        raise RuntimeError("Expected 1,200 rows in each frozen artifact")
    if len({x["task_id"] for x in inputs}) != 600 or len({x["task_id"] for x in outputs}) != 600:
        raise RuntimeError("Expected 600 unique tasks")
    if Counter(x["condition"] for x in inputs) != Counter({c: 600 for c in CONDITIONS}):
        raise RuntimeError("Unexpected frozen input conditions")
    if Counter(x["condition"] for x in outputs) != Counter({c: 600 for c in CONDITIONS}):
        raise RuntimeError("Unexpected frozen output conditions")
    if any(x.get("error") for x in outputs) or any(not str(x.get("generated_answer", "")).strip() for x in outputs):
        raise RuntimeError("Frozen generation errors or empty answers detected")

    input_index = {(x["task_id"], x["condition"]): x for x in inputs}
    output_index = {(x["task_id"], x["condition"]): x for x in outputs}
    if len(input_index) != 1200 or len(output_index) != 1200 or set(input_index) != set(output_index):
        raise RuntimeError("Frozen input/output key mismatch")
    copied_fields = ("question", "reference_answer", "context_text", "context_doc_ids", "task_type", "target_lecture")
    for key in input_index:
        for field in copied_fields:
            if input_index[key][field] != output_index[key][field]:
                raise RuntimeError(f"Frozen input/output mismatch for {key} field {field}")

    grouped: dict[str, dict[str, dict]] = defaultdict(dict)
    for row in outputs:
        grouped[row["task_id"]][row["condition"]] = row
    if any(set(pair) != set(CONDITIONS) for pair in grouped.values()):
        raise RuntimeError("Every task must have exactly the two expected conditions")

    rng = random.Random(SEED)
    primary_rows, swapped_rows, mapping_rows = [], [], []
    for task_id in sorted(grouped):
        pair = grouped[task_id]
        primary_a = CONDITIONS[0] if rng.random() < 0.5 else CONDITIONS[1]
        primary_b = CONDITIONS[1] if primary_a == CONDITIONS[0] else CONDITIONS[0]

        def make_row(pass_name: str, a_condition: str, b_condition: str) -> dict:
            a, b = pair[a_condition], pair[b_condition]
            if a["question"] != b["question"] or a["reference_answer"] != b["reference_answer"]:
                raise RuntimeError(f"Question/reference mismatch within task {task_id}")
            return {
                "judge_item_id": f"{pass_name}::{task_id}",
                "task_id": task_id,
                "question": a["question"],
                "reference_answer": a["reference_answer"],
                "evidence_A": a["context_text"],
                "answer_A": a["generated_answer"],
                "evidence_B": b["context_text"],
                "answer_B": b["generated_answer"],
            }

        primary_rows.append(make_row("primary", primary_a, primary_b))
        swapped_rows.append(make_row("swapped", primary_b, primary_a))
        exemplar = pair[primary_a]
        mapping_rows.append({
            "task_id": task_id,
            "task_type": exemplar["task_type"],
            "target_lecture": exemplar["target_lecture"],
            "primary_A_condition": primary_a,
            "primary_B_condition": primary_b,
            "swapped_A_condition": primary_b,
            "swapped_B_condition": primary_a,
            "primary_A_context_doc_ids_json": json.dumps(pair[primary_a]["context_doc_ids"]),
            "primary_B_context_doc_ids_json": json.dumps(pair[primary_b]["context_doc_ids"]),
            "randomization_seed": SEED,
        })

    primary_path = out / "clean_inputs_primary_600.jsonl"
    swapped_path = out / "clean_inputs_swapped_600.jsonl"
    mapping_path = out / "blinded_condition_mapping.csv"
    for path in (primary_path, swapped_path, mapping_path):
        if path.exists():
            raise RuntimeError(f"Refusing to overwrite existing output: {path}")
    write_jsonl(primary_path, primary_rows)
    write_jsonl(swapped_path, swapped_rows)
    with mapping_path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(mapping_rows[0]))
        writer.writeheader()
        writer.writerows(mapping_rows)

    # Re-read the serialized artifacts and compare every substantive string to its exact frozen source.
    serialized = {"primary": read_jsonl(primary_path), "swapped": read_jsonl(swapped_path)}
    mapping = {x["task_id"]: x for x in mapping_rows}
    comparisons = 0
    modifications = []
    for pass_name, rows in serialized.items():
        for row in rows:
            m = mapping[row["task_id"]]
            a_condition, b_condition = m[f"{pass_name}_A_condition"], m[f"{pass_name}_B_condition"]
            a, b = grouped[row["task_id"]][a_condition], grouped[row["task_id"]][b_condition]
            expected = {
                "question": a["question"], "reference_answer": a["reference_answer"],
                "evidence_A": a["context_text"], "answer_A": a["generated_answer"],
                "evidence_B": b["context_text"], "answer_B": b["generated_answer"],
            }
            for field in SUBSTANTIVE_FIELDS:
                comparisons += 1
                if row[field] != expected[field]:
                    modifications.append({"pass": pass_name, "task_id": row["task_id"], "field": field})
    if modifications:
        raise RuntimeError(f"Substantive text integrity failure: {len(modifications)} fields differ")

    integrity = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "frozen_generation_input_path": str(input_path),
        "frozen_generation_input_sha256": EXPECTED_INPUT_SHA,
        "frozen_generation_output_path": str(output_path),
        "frozen_generation_output_sha256": EXPECTED_OUTPUT_SHA,
        "frozen_input_rows": len(inputs),
        "frozen_output_rows": len(outputs),
        "unique_tasks": len(grouped),
        "condition_counts": dict(Counter(x["condition"] for x in outputs)),
        "generation_errors": 0,
        "empty_answers": 0,
        "randomization_seed": SEED,
        "primary_rows": len(primary_rows),
        "swapped_rows": len(swapped_rows),
        "substantive_fields_compared": list(SUBSTANTIVE_FIELDS),
        "substantive_field_comparisons": comparisons,
        "modified_substantive_fields": 0,
        "modifications": [],
        "clean_primary_sha256": sha256(primary_path),
        "clean_swapped_sha256": sha256(swapped_path),
        "mapping_sha256": sha256(mapping_path),
    }
    (out / "blinding_integrity.json").write_text(json.dumps(integrity, indent=2), encoding="utf-8")
    source_prompt = Path(__file__).resolve().parents[1] / "common_judge_prompt.txt"
    output_prompt = out / "common_judge_prompt.txt"
    if not output_prompt.exists():
        output_prompt.write_text(source_prompt.read_text(encoding="utf-8"), encoding="utf-8")
    prompt_sha = sha256(output_prompt)
    report = f"""# Clean blinding integrity report

## Result

`modified_substantive_fields = 0` — PASS.

The two clean datasets contain 600 primary and 600 exact-swapped tasks. For every task and orientation, the question, reference answer, supplied evidence text, and generated Answer A/B were copied byte-for-byte at the decoded string level from the original frozen generation artifacts. No word substitutions or other text sanitization were performed. Blinding is achieved only by withholding condition labels from judge-visible prompts; the hidden mapping is stored separately.

## Verified frozen artifacts

- Generation inputs: `{input_path}` — `{EXPECTED_INPUT_SHA}` — 1,200 rows.
- Generation outputs: `{output_path}` — `{EXPECTED_OUTPUT_SHA}` — 1,200 rows, 600 tasks, 600 rows per condition, zero errors, zero empty answers.

## Integrity counts

- Substantive string comparisons: {comparisons:,}
- Modified substantive fields: 0
- Primary clean-input SHA256: `{integrity['clean_primary_sha256']}`
- Swapped clean-input SHA256: `{integrity['clean_swapped_sha256']}`
- Hidden mapping SHA256: `{integrity['mapping_sha256']}`
- Common rubric SHA256: `{prompt_sha}`

Judge runners construct the visible request from only: Student question, Reference answer, Evidence A, Answer A, Evidence B, and Answer B. Task identifiers and the hidden condition mapping are not sent to either judge.
"""
    (out / "BLINDING_INTEGRITY_REPORT.md").write_text(report, encoding="utf-8")
    print("CLEAN_BLINDING_INTEGRITY = PASS modified_substantive_fields=0")


if __name__ == "__main__":
    main()
