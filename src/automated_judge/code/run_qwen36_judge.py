#!/usr/bin/env python3
"""Run both orientations of the clean blinded Qwen3.6 judge with vLLM."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
# The installed vLLM build defaults DeepGEMM warmup on even when the optional
# DeepGEMM package is unavailable. Disable that optional path before vLLM import.
os.environ.setdefault("VLLM_USE_DEEP_GEMM", "0")
os.environ.setdefault("VLLM_MOE_USE_DEEP_GEMM", "0")
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch
import transformers
import vllm
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.sampling_params import StructuredOutputsParams


MODEL_ID = "Qwen/Qwen3.6-35B-A3B"
REVISION = "995ad96eacd98c81ed38be0c5b274b04031597b0"
EXPECTED_SNAPSHOT = f"models--Qwen--Qwen3.6-35B-A3B/snapshots/{REVISION}"
SEED = 20260919
MAX_TOKENS = 220
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
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def validate(text: str) -> tuple[dict | None, str | None]:
    try:
        parsed = json.loads(text)
    except Exception as exc:
        return None, f"json_parse:{type(exc).__name__}:{exc}"
    required = set(SCHEMA["required"])
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


def visible_user_payload(row: dict) -> str:
    # Deliberately excludes task ID, task metadata, condition labels, and context IDs.
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--model-path", required=True, type=Path)
    parser.add_argument("--gpu-index", required=True, type=int)
    parser.add_argument("--tensor-parallel-size", default=1, type=int)
    args = parser.parse_args()
    out = args.output_root.resolve()
    qwen_out = out / "qwen36"
    model_path = args.model_path.resolve()
    if not str(model_path).endswith(EXPECTED_SNAPSHOT):
        raise RuntimeError(f"Unexpected model snapshot: {model_path}")
    if not (model_path / "model.safetensors.index.json").is_file():
        raise RuntimeError("Pinned model snapshot is incomplete")
    ref_path = model_path.parents[1] / "refs/main"
    if not ref_path.is_file() or ref_path.read_text().strip() != REVISION:
        raise RuntimeError("Local Qwen3.6 refs/main does not match pinned revision")

    raw_paths = {name: qwen_out / f"qwen36_{name}_raw_600.jsonl" for name in ("primary", "swapped")}
    for path in raw_paths.values():
        if path.exists():
            raise RuntimeError(f"Refusing to overwrite existing raw judgment output: {path}")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_index)
    prompt_path = out / "common_judge_prompt.txt"
    system_prompt = prompt_path.read_text(encoding="utf-8").strip()
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True)

    prepared: dict[str, tuple[list[dict], list[str]]] = {}
    prompt_lengths = []
    for pass_name in ("primary", "swapped"):
        rows = read_jsonl(out / f"clean_inputs_{pass_name}_600.jsonl")
        if len(rows) != 600 or len({x["task_id"] for x in rows}) != 600:
            raise RuntimeError(f"Expected exactly 600 unique {pass_name} items")
        formatted = [tokenizer.apply_chat_template(
            [{"role": "system", "content": system_prompt},
             {"role": "user", "content": visible_user_payload(row)}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        ) for row in rows]
        pass_lengths = [len(tokenizer.encode(text, add_special_tokens=False)) for text in formatted]
        prompt_lengths.extend(pass_lengths)
        if max(pass_lengths) + MAX_TOKENS > 8192:
            raise RuntimeError(f"Prompt exceeds configured context in {pass_name}: {max(pass_lengths)}")
        prepared[pass_name] = (rows, formatted)

    llm = LLM(
        model=str(model_path),
        tokenizer=str(model_path),
        runner="generate",
        trust_remote_code=False,
        tensor_parallel_size=args.tensor_parallel_size,
        dtype="bfloat16",
        gpu_memory_utilization=0.95,
        max_model_len=8192,
        max_num_seqs=48,
        seed=SEED,
        language_model_only=True,
        moe_backend="triton",
        gdn_prefill_backend="triton",
        enforce_eager=True,
        limit_mm_per_prompt={"image": 0, "video": 0},
    )
    params = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        top_k=0,
        max_tokens=MAX_TOKENS,
        seed=SEED,
        structured_outputs=StructuredOutputsParams(json=SCHEMA),
    )
    pass_metadata = {}
    for pass_name in ("primary", "swapped"):
        rows, formatted = prepared[pass_name]
        outputs = llm.generate(formatted, params, use_tqdm=True)
        records, failure_indices = [], []
        for row, request_output in zip(rows, outputs):
            candidate = request_output.outputs[0]
            parsed, error = validate(candidate.text)
            record = {
                "judge_item_id": row["judge_item_id"],
                "pass": pass_name,
                "task_id": row["task_id"],
                "attempts": [{
                    "attempt": 1,
                    "raw_output": candidate.text,
                    "validation_error": error,
                    "finish_reason": candidate.finish_reason,
                    "prompt_tokens": len(request_output.prompt_token_ids),
                    "output_tokens": len(candidate.token_ids),
                }],
                "selected_attempt": 1 if error is None else None,
                "parsed": parsed,
            }
            records.append(record)
            if error is not None:
                failure_indices.append(len(records) - 1)

        # One documented deterministic constrained retry for each malformed/overlong result.
        for index in failure_indices:
            retry_prompt = formatted[index] + "\nReturn exactly the required JSON. The brief_reason must contain at most 50 words."
            retry_output = llm.generate([retry_prompt], params, use_tqdm=False)[0]
            candidate = retry_output.outputs[0]
            parsed, error = validate(candidate.text)
            records[index]["attempts"].append({
                "attempt": 2,
                "raw_output": candidate.text,
                "validation_error": error,
                "finish_reason": candidate.finish_reason,
                "prompt_tokens": len(retry_output.prompt_token_ids),
                "output_tokens": len(candidate.token_ids),
            })
            records[index]["selected_attempt"] = 2 if error is None else None
            records[index]["parsed"] = parsed

        remaining = [record for record in records if record["parsed"] is None]
        with raw_paths[pass_name].open("x", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        pass_metadata[pass_name] = {
            "items": len(records),
            "initial_parser_failures": len(failure_indices),
            "retry_count": len(failure_indices),
            "remaining_parser_failures": len(remaining),
            "input_sha256": sha256(out / f"clean_inputs_{pass_name}_600.jsonl"),
            "raw_output_sha256": sha256(raw_paths[pass_name]),
            "completed_utc": datetime.now(timezone.utc).isoformat(),
        }
        (qwen_out / "qwen36_inference_metadata.json").write_text(json.dumps({
            "generator_model": "Qwen/Qwen3-8B",
            "judge_model": MODEL_ID,
            "judge_revision_sha": REVISION,
            "local_snapshot_path": str(model_path),
            "inference_backend": f"vLLM {vllm.__version__}",
            "python_version": sys.version,
            "platform": platform.platform(),
            "torch_version": torch.__version__,
            "cuda_runtime_version": torch.version.cuda,
            "transformers_version": transformers.__version__,
            "vllm_version": vllm.__version__,
            "gpu_model": torch.cuda.get_device_name(0),
            "selected_physical_gpu_ids": [args.gpu_index + i for i in range(args.tensor_parallel_size)],
            "tensor_parallel_size": args.tensor_parallel_size,
            "language_model_only": True,
            "moe_backend": "triton",
            "gdn_prefill_backend": "triton",
            "enforce_eager": True,
            "optional_deepgemm_disabled": True,
            "non_thinking_setting": "tokenizer.apply_chat_template(enable_thinking=False)",
            "structured_decoding": "vLLM StructuredOutputsParams(json=SCHEMA)",
            "decoding_parameters": {"temperature": 0.0, "do_sample": False, "top_p": 1.0,
                                    "top_k": 0, "max_tokens": MAX_TOKENS, "seed": SEED,
                                    "dtype": "bfloat16", "max_model_len": 8192},
            "common_prompt_sha256": sha256(prompt_path),
            "prompt_tokens_min": min(prompt_lengths),
            "prompt_tokens_max": max(prompt_lengths),
            "passes": pass_metadata,
        }, indent=2), encoding="utf-8")
        if remaining:
            raise RuntimeError(f"Qwen3.6 parser gate failed for {len(remaining)} {pass_name} items")
        print(f"QWEN36_{pass_name.upper()} = PASS items={len(records)} retries={len(failure_indices)}")


if __name__ == "__main__":
    main()
