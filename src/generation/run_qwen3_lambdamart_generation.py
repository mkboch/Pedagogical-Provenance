#!/usr/bin/env python3

# Public path-portable copy of the frozen analysis source.
# Frozen analysis source SHA256: ba5414044f3285e0374ca7ad77bbc91308a4b2cc83fdecbe1bad60f809951782

from pathlib import Path
import hashlib
import json
import os
import platform
import time

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)


ROOT = Path(__file__).resolve().parents[2]

OUT = (
    ROOT
    / "artifacts/jiis_final_experiments_20260918"
)

INPUT = (
    OUT
    / "lambdamart_qwen3_generation_inputs_1200.jsonl"
)

JOURNAL = (
    OUT
    / "qwen3_lambdamart_generation_1200_journal.jsonl"
)

FINAL = (
    OUT
    / "qwen3_lambdamart_generation_1200_final.jsonl"
)

META = (
    OUT
    / "QWEN3_LAMBDAMART_GENERATION_METADATA.json"
)

MODEL = "Qwen/Qwen3-8B"
MAX_NEW_TOKENS = 256



OUT.mkdir(parents=True, exist_ok=True)

def sha256(path):
    h = hashlib.sha256()

    with path.open("rb") as f:
        for b in iter(
            lambda: f.read(
                1024 * 1024
            ),
            b"",
        ):
            h.update(b)

    return h.hexdigest()


def read_jsonl(path):
    rows = []

    if not path.exists():
        return rows

    with path.open(
        "r",
        encoding="utf-8",
    ) as f:
        for line in f:
            if line.strip():
                rows.append(
                    json.loads(line)
                )

    return rows


def append_jsonl(path, row):
    with path.open(
        "a",
        encoding="utf-8",
    ) as f:
        f.write(
            json.dumps(
                row,
                ensure_ascii=False,
            )
            + "\n"
        )

        f.flush()
        os.fsync(
            f.fileno()
        )


inputs = read_jsonl(INPUT)

if len(inputs) != 1200:
    raise RuntimeError(
        f"Expected 1200 prompts, got {len(inputs)}"
    )

if len(
    {
        r["run_id"]
        for r in inputs
    }
) != 1200:
    raise RuntimeError(
        "Generation input run IDs are not unique."
    )

existing_rows = read_jsonl(
    JOURNAL
)

existing = {
    r["run_id"]: r
    for r in existing_rows
    if r.get("run_id")
    and not r.get("error")
    and str(
        r.get(
            "generated_answer",
            "",
        )
    ).strip()
}

missing = [
    r
    for r in inputs
    if r["run_id"]
    not in existing
]

print("=" * 100)
print("QWEN3 LAMBDAMART END-TO-END GENERATION")
print("=" * 100)
print("Model:", MODEL)
print("Total:", len(inputs))
print("Already successful:", len(existing))
print("Missing:", len(missing))
print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"))
print("torch:", torch.__version__)
print("cuda:", torch.version.cuda)

if not missing:
    print("Nothing to generate.")
else:
    print("Loading tokenizer...")

    tokenizer = (
        AutoTokenizer
        .from_pretrained(
            MODEL,
            trust_remote_code=True,
            local_files_only=True,
        )
    )

    print("Loading model...")

    model = (
        AutoModelForCausalLM
        .from_pretrained(
            MODEL,
            dtype=(
                torch.bfloat16
                if torch.cuda.is_available()
                else torch.float32
            ),
            device_map="auto",
            trust_remote_code=True,
            local_files_only=True,
        )
    )

    model.eval()

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = (
            tokenizer.eos_token_id
        )

    for index, rec in enumerate(
        missing,
        start=1,
    ):
        prompt = str(
            rec["prompt"]
        )

        messages = [
            {
                "role": "user",
                "content": prompt,
            }
        ]

        try:
            serialized = (
                tokenizer
                .apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
            )
        except TypeError:
            serialized = (
                tokenizer
                .apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            )

        encoded = tokenizer(
            serialized,
            return_tensors="pt",
        )

        device = next(
            model.parameters()
        ).device

        encoded = {
            k: v.to(device)
            for k, v in encoded.items()
        }

        input_tokens = int(
            encoded[
                "input_ids"
            ].shape[1]
        )

        t0 = time.time()

        try:
            with torch.inference_mode():
                generated = (
                    model.generate(
                        **encoded,
                        max_new_tokens=MAX_NEW_TOKENS,
                        do_sample=False,
                        use_cache=True,
                        pad_token_id=(
                            tokenizer
                            .pad_token_id
                        ),
                        eos_token_id=(
                            tokenizer
                            .eos_token_id
                        ),
                    )
                )

            new_ids = generated[
                0,
                input_tokens:,
            ]

            answer = (
                tokenizer.decode(
                    new_ids,
                    skip_special_tokens=True,
                )
                .strip()
            )

            row = dict(rec)

            row.update(
                {
                    "model_key":
                        "qwen3_8b",
                    "model_name":
                        MODEL,
                    "generated_answer":
                        answer,
                    "input_tokens":
                        input_tokens,
                    "output_tokens":
                        int(
                            new_ids.numel()
                        ),
                    "latency_sec":
                        float(
                            time.time()
                            - t0
                        ),
                    "max_new_tokens":
                        MAX_NEW_TOKENS,
                    "do_sample":
                        False,
                    "error":
                        None,
                }
            )

        except Exception as e:
            row = dict(rec)

            row.update(
                {
                    "model_key":
                        "qwen3_8b",
                    "model_name":
                        MODEL,
                    "generated_answer":
                        "",
                    "latency_sec":
                        float(
                            time.time()
                            - t0
                        ),
                    "max_new_tokens":
                        MAX_NEW_TOKENS,
                    "do_sample":
                        False,
                    "error":
                        repr(e),
                }
            )

        append_jsonl(
            JOURNAL,
            row,
        )

        status = (
            "OK"
            if not row.get("error")
            else "ERROR"
        )

        print(
            f"[{index}/{len(missing)}] "
            f"{row['run_id']} "
            f"{status} "
            f"{row.get('latency_sec', 0):.2f}s"
        )


rows = read_jsonl(
    JOURNAL
)

latest = {}

for r in rows:
    latest[
        r["run_id"]
    ] = r

final = [
    latest[
        rec["run_id"]
    ]
    for rec in inputs
    if rec["run_id"]
    in latest
]

errors = [
    r
    for r in final
    if r.get("error")
]

empty = [
    r
    for r in final
    if not str(
        r.get(
            "generated_answer",
            "",
        )
    ).strip()
]

if len(final) != 1200:
    raise RuntimeError(
        f"Final generation rows={len(final)}; expected 1200"
    )

if errors:
    raise RuntimeError(
        f"Generation contains {len(errors)} errors."
    )

if empty:
    raise RuntimeError(
        f"Generation contains {len(empty)} empty answers."
    )

with FINAL.open(
    "w",
    encoding="utf-8",
) as f:
    for r in final:
        f.write(
            json.dumps(
                r,
                ensure_ascii=False,
            )
            + "\n"
        )


meta = {
    "model":
        MODEL,
    "max_new_tokens":
        MAX_NEW_TOKENS,
    "do_sample":
        False,
    "enable_thinking":
        False,
    "rows":
        len(final),
    "input_sha256":
        sha256(INPUT),
    "final_sha256":
        sha256(FINAL),
    "torch":
        torch.__version__,
    "cuda":
        torch.version.cuda,
    "python":
        platform.python_version(),
}

META.write_text(
    json.dumps(
        meta,
        indent=2,
    )
    + "\n",
    encoding="utf-8",
)

(
    OUT
    / "QWEN3_LAMBDAMART_GENERATION_COMPLETE.txt"
).write_text(
    "PASS\n",
    encoding="utf-8",
)

print()
print("=" * 100)
print("QWEN3 GENERATION COMPLETE")
print("=" * 100)
print("Rows:", len(final))
print("Errors:", len(errors))
print("Empty:", len(empty))
print("Final SHA256:", sha256(FINAL))
print("QWEN3_LAMBDAMART_GENERATION_COMPLETE = PASS")
