#!/usr/bin/env python3

from pathlib import Path
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time


ROOT = Path(__file__).resolve().parents[2]

FREEZE = (
    ROOT
    / "artifacts/matched_retrieval"
)

OUTROOT = (
    ROOT
    / "artifacts/matched_generation"
)

LOCKFILE = (
    OUTROOT
    / "GENERATION_INPUT_LOCK.json"
)

INPUTS = (
    FREEZE
    / "frozen_position_contexts_and_prompts_2000.jsonl"
)

MODEL_ARTIFACTS = {
    "qwen3_8b": {
        "model_name": "Qwen/Qwen3-8B",
        "chat_mode": "qwen",
    },
    "qwen2_5_7b": {
        "model_name": "Qwen/Qwen2.5-7B-Instruct",
        "chat_mode": "qwen",
    },
    "mistral_7b": {
        "model_name": "mistralai/Mistral-7B-Instruct-v0.3",
        "chat_mode": "mistral",
    },
}


# =============================================================================
# BASIC HELPERS
# =============================================================================

def sha256_file(path):
    h = hashlib.sha256()

    with path.open("rb") as f:
        while True:
            b = f.read(
                1024 * 1024
            )

            if not b:
                break

            h.update(b)

    return h.hexdigest()


def sha256_text(text):
    return hashlib.sha256(
        str(text).encode(
            "utf-8"
        )
    ).hexdigest()


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
                    json.loads(
                        line
                    )
                )

    return rows


def append_jsonl(path, row):
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

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


def atomic_jsonl(path, rows):
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fd, temp_name = tempfile.mkstemp(
        prefix=path.name + ".",
        suffix=".tmp",
        dir=str(
            path.parent
        ),
    )

    os.close(fd)

    temp = Path(
        temp_name
    )

    try:
        with temp.open(
            "w",
            encoding="utf-8",
        ) as f:

            for row in rows:
                f.write(
                    json.dumps(
                        row,
                        ensure_ascii=False,
                    )
                    + "\n"
                )

        os.replace(
            temp,
            path,
        )

    finally:
        if temp.exists():
            temp.unlink()


def parse_list(x):
    if isinstance(
        x,
        list,
    ):
        return x

    if x is None:
        return []

    s = str(x).strip()

    if not s:
        return []

    try:
        obj = json.loads(s)

        if isinstance(
            obj,
            list,
        ):
            return obj

    except Exception:
        pass

    raise RuntimeError(
        f"Could not parse JSON list: {x!r}"
    )


# =============================================================================
# INPUT LOCK / PROTOCOL VALIDATION
# =============================================================================

def validate_lock():
    if not LOCKFILE.exists():
        raise RuntimeError(
            f"Missing lock file: {LOCKFILE}"
        )

    lock = json.loads(
        LOCKFILE.read_text(
            encoding="utf-8"
        )
    )

    for name, meta in lock[
        "frozen_files"
    ].items():

        p = Path(
            meta["path"]
        )

        if not p.exists():
            raise RuntimeError(
                f"Locked file missing: {p}"
            )

        actual = sha256_file(
            p
        )

        expected = meta[
            "sha256"
        ]

        if actual != expected:
            raise RuntimeError(
                f"LOCK FAILURE for {name}\n"
                f"expected={expected}\n"
                f"actual  ={actual}\n"
                f"path={p}"
            )

    return lock


def validate_inputs():
    rows = read_jsonl(
        INPUTS
    )

    if len(rows) != 2000:
        raise RuntimeError(
            f"Expected 2000 frozen rows, got {len(rows)}"
        )

    seen = set()
    counts = {
        "bm25": 0,
        "rrf": 0,
    }

    for row in rows:

        tid = str(
            row["task_id"]
        )

        method = str(
            row[
                "retrieval_method"
            ]
        )

        if method not in counts:
            raise RuntimeError(
                f"Unexpected retrieval method: {method}"
            )

        key = (
            tid,
            method,
        )

        if key in seen:
            raise RuntimeError(
                f"Duplicate task/method: {key}"
            )

        seen.add(
            key
        )

        counts[
            method
        ] += 1

        prompt = str(
            row["prompt"]
        )

        expected_hash = str(
            row[
                "prompt_sha256"
            ]
        )

        actual_hash = sha256_text(
            prompt
        )

        if actual_hash != expected_hash:
            raise RuntimeError(
                f"Prompt hash mismatch: {key}"
            )

        context_ids = parse_list(
            row[
                "context_doc_ids"
            ]
        )

        if not (
            1
            <= len(context_ids)
            <= 3
        ):
            raise RuntimeError(
                f"Invalid context count {len(context_ids)}: {key}"
            )

    if counts != {
        "bm25": 1000,
        "rrf": 1000,
    }:
        raise RuntimeError(
            f"Unexpected method counts: {counts}"
        )

    return rows, counts


# =============================================================================
# ORIGINAL MODEL ID + SOURCE SERIALIZATION AUDIT
# =============================================================================

def get_original_model_id(
    model_key,
):
    return MODEL_ARTIFACTS[
        model_key
    ][
        "model_name"
    ]


def validate_source_serialization(
    model_key,
):
    cfg = MODEL_ARTIFACTS[
        model_key
    ]

    return {
        "source":
            "src/generation/run_matched_global_position_generation.py",

        "sha256":
            sha256_file(
                Path(__file__)
            ),

        "chat_mode":
            cfg[
                "chat_mode"
            ],
    }


# =============================================================================
# GPU CHECK
# =============================================================================

def gpu_status():
    try:
        cmd = [
            "nvidia-smi",
            "--query-gpu=index,name,memory.total,memory.used,memory.free",
            "--format=csv,noheader,nounits",
        ]

        text = subprocess.check_output(
            cmd,
            text=True,
        )

        rows = []

        for line in text.strip().splitlines():

            parts = [
                x.strip()
                for x in line.split(",")
            ]

            if len(parts) != 5:
                continue

            rows.append(
                {
                    "index":
                        int(
                            parts[0]
                        ),

                    "name":
                        parts[1],

                    "total_mib":
                        int(
                            parts[2]
                        ),

                    "used_mib":
                        int(
                            parts[3]
                        ),

                    "free_mib":
                        int(
                            parts[4]
                        ),
                }
            )

        return rows

    except Exception as e:
        return [
            {
                "error":
                    repr(e)
            }
        ]


# =============================================================================
# CHAT SERIALIZATION
# =============================================================================

def serialize_prompt(
    tokenizer,
    model_key,
    prompt,
):
    messages = [
        {
            "role":
                "user",

            "content":
                prompt,
        }
    ]

    mode = MODEL_ARTIFACTS[
        model_key
    ][
        "chat_mode"
    ]

    if mode == "qwen":

        try:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )

        except TypeError:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

    if mode == "mistral":

        try:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

        except Exception:
            return (
                f"<s>[INST] "
                f"{prompt} "
                f"[/INST]"
            )

    raise RuntimeError(
        f"Unknown chat mode: {mode}"
    )


# =============================================================================
# JOURNAL / COMPACT OUTPUT
# =============================================================================

def output_paths(
    model_key,
):
    model_dir = (
        OUTROOT
        / model_key
    )

    model_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    return {
        "journal":
            model_dir
            / (
                f"{model_key}"
                "_matched_position_2000_journal.jsonl"
            ),

        "compact":
            model_dir
            / (
                f"{model_key}"
                "_matched_position_2000.jsonl"
            ),

        "metadata":
            model_dir
            / (
                f"{model_key}"
                "_matched_position_metadata.json"
            ),
    }


def latest_state(
    journal,
    compact,
):
    latest = {}

    for path in [
        compact,
        journal,
    ]:

        if not path.exists():
            continue

        for row in read_jsonl(
            path
        ):

            rid = row.get(
                "run_id"
            )

            if rid:
                latest[
                    rid
                ] = row

    return latest


def is_success(
    row,
):
    if not row:
        return False

    if row.get(
        "error"
    ):
        return False

    return bool(
        str(
            row.get(
                "generated_answer",
                "",
            )
        ).strip()
    )


def compact_state(
    path,
    state,
):
    rows = [
        state[k]
        for k in sorted(
            state
        )
    ]

    atomic_jsonl(
        path,
        rows,
    )


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        required=True,
        choices=sorted(
            MODEL_ARTIFACTS
        ),
    )

    parser.add_argument(
        "--preflight",
        action="store_true",
    )

    parser.add_argument(
        "--min-free-mib",
        type=int,
        default=26000,
    )

    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=25,
    )

    args = parser.parse_args()

    lock = validate_lock()

    rows, counts = validate_inputs()

    model_id = get_original_model_id(
        args.model
    )

    source_meta = validate_source_serialization(
        args.model
    )

    gpu = gpu_status()

    print(
        "=" * 105
    )

    print(
        "MATCHED POSITION GENERATION — PREFLIGHT"
    )

    print(
        "=" * 105
    )

    print(
        "Model key:",
        args.model,
    )

    print(
        "Model ID:",
        model_id,
    )

    print(
        "Frozen inputs:",
        len(rows),
    )

    print(
        "Method counts:",
        counts,
    )

    print(
        "Input lock:",
        LOCKFILE,
    )

    print(
        "Serialization implementation:",
        source_meta[
            "source"
        ],
    )

    print(
        "Generation script SHA256:",
        source_meta[
            "sha256"
        ],
    )

    print(
        "Chat mode:",
        source_meta[
            "chat_mode"
        ],
    )

    print(
        "Decoding: max_new_tokens=256, do_sample=False"
    )

    print()

    print(
        "Physical GPU state:"
    )

    for x in gpu:
        print(
            " ",
            x,
        )

    print()

    print(
        "CUDA_VISIBLE_DEVICES:",
        os.environ.get(
            "CUDA_VISIBLE_DEVICES",
            "<not set>",
        ),
    )

    if args.preflight:

        print()

        print(
            "PROTOCOL_PREFLIGHT = PASSED"
        )

        print(
            "GENERATION_STARTED = False"
        )

        return

    # Import GPU libraries only after all protocol checks pass.
    import torch

    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
    )

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available in this Python environment."
        )

    free_bytes, total_bytes = (
        torch.cuda.mem_get_info(
            0
        )
    )

    free_mib = (
        free_bytes
        / 1024
        / 1024
    )

    print(
        f"Visible CUDA device free memory: "
        f"{free_mib:.0f} MiB"
    )

    if free_mib < args.min_free_mib:

        raise RuntimeError(
            f"REFUSING TO LOAD MODEL: "
            f"free GPU memory {free_mib:.0f} MiB "
            f"< required safety threshold "
            f"{args.min_free_mib} MiB."
        )

    paths = output_paths(
        args.model
    )

    state = latest_state(
        paths[
            "journal"
        ],
        paths[
            "compact"
        ],
    )

    success_before = sum(
        is_success(x)
        for x in state.values()
    )

    print(
        "Already successful:",
        success_before,
        "/ 2000",
    )

    missing = []

    for row in rows:

        run_id = (
            f"{row['system_name']}"
            f"::{row['task_id']}"
        )

        if not is_success(
            state.get(
                run_id
            )
        ):
            missing.append(
                row
            )

    print(
        "Missing:",
        len(missing),
    )

    metadata = {
        "protocol":
            "matched_position_generation",

        "model_key":
            args.model,

        "model_name":
            model_id,

        "input_lock":
            lock,

        "reference_generation_source":
            source_meta,

        "max_new_tokens":
            256,

        "do_sample":
            False,

        "chat_serialization":
            source_meta[
                "chat_mode"
            ],

        "input_count":
            2000,

        "started_at":
            time.strftime(
                "%Y-%m-%d %H:%M:%S"
            ),

        "python":
            sys.executable,

        "torch_version":
            torch.__version__,

        "cuda_version":
            torch.version.cuda,
    }

    paths[
        "metadata"
    ].write_text(
        json.dumps(
            metadata,
            indent=2,
        ),
        encoding="utf-8",
    )

    if not missing:

        compact_state(
            paths[
                "compact"
            ],
            state,
        )

        print(
            "MODEL COMPLETE: 2000/2000 already successful."
        )

        return

    print()
    print(
        "Loading tokenizer:",
        model_id,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        trust_remote_code=True,
    )

    print(
        "Loading model:",
        model_id,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=(
            torch.bfloat16
            if torch.cuda.is_available()
            else torch.float32
        ),
        device_map=None,
        trust_remote_code=True,
    )

    model = model.to(
        "cuda"
    )

    model.eval()

    print(
        "Model loaded."
    )

    print()

    generated_this_run = 0

    for idx, rec in enumerate(
        missing,
        start=1,
    ):

        run_id = (
            f"{rec['system_name']}"
            f"::{rec['task_id']}"
        )

        start = time.time()

        error = ""

        answer = ""

        input_tokens = None

        output_tokens = None

        try:
            serialized = serialize_prompt(
                tokenizer,
                args.model,
                rec[
                    "prompt"
                ],
            )

            model_inputs = tokenizer(
                serialized,
                return_tensors="pt",
            )

            model_inputs = {
                k:
                    v.to(
                        "cuda"
                    )
                for k, v
                in model_inputs.items()
            }

            input_tokens = int(
                model_inputs[
                    "input_ids"
                ].shape[-1]
            )

            with torch.inference_mode():

                outputs = model.generate(
                    **model_inputs,
                    max_new_tokens=256,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                )

            new_tokens = outputs[
                0
            ][
                model_inputs[
                    "input_ids"
                ].shape[-1]:
            ]

            output_tokens = int(
                new_tokens.shape[-1]
            )

            answer = tokenizer.decode(
                new_tokens,
                skip_special_tokens=True,
            ).strip()

        except Exception as e:
            error = repr(
                e
            )

        item = {
            "protocol":
                "matched_position_generation",

            "model_name":
                model_id,

            "model_key":
                args.model,

            "run_id":
                run_id,

            "system_name":
                rec[
                    "system_name"
                ],

            "original_prompt_system_name":
                rec[
                    "original_prompt_system_name"
                ],

            "retrieval_method":
                rec[
                    "retrieval_method"
                ],

            "task_id":
                rec[
                    "task_id"
                ],

            "task_type":
                rec[
                    "task_type"
                ],

            "target_doc_id":
                rec[
                    "target_doc_id"
                ],

            "gold_evidence_doc_ids":
                parse_list(
                    rec[
                        "gold_evidence_doc_ids"
                    ]
                ),

            "context_doc_ids":
                parse_list(
                    rec[
                        "context_doc_ids"
                    ]
                ),

            "question":
                rec[
                    "question"
                ],

            "reference_answer":
                rec[
                    "reference_answer"
                ],

            "prompt_sha256":
                rec[
                    "prompt_sha256"
                ],

            "generated_answer":
                answer,

            "error":
                error,

            "input_tokens":
                input_tokens,

            "output_tokens":
                output_tokens,

            "latency_sec":
                round(
                    time.time()
                    - start,
                    3,
                ),
        }

        append_jsonl(
            paths[
                "journal"
            ],
            item,
        )

        state[
            run_id
        ] = item

        if not error:
            generated_this_run += 1

        status = (
            "OK"
            if not error
            else "ERROR"
        )

        total_success = sum(
            is_success(x)
            for x in state.values()
        )

        print(
            f"[{idx}/{len(missing)}] "
            f"{status} "
            f"{run_id} "
            f"lat={item['latency_sec']:.3f}s "
            f"out_tokens={output_tokens} "
            f"total_success={total_success}/2000",
            flush=True,
        )

        if (
            idx
            % args.checkpoint_every
            == 0
        ):
            compact_state(
                paths[
                    "compact"
                ],
                state,
            )

    compact_state(
        paths[
            "compact"
        ],
        state,
    )

    final_success = sum(
        is_success(x)
        for x in state.values()
    )

    final_errors = sum(
        not is_success(x)
        for x in state.values()
    )

    print()
    print(
        "=" * 105
    )

    print(
        "MODEL GENERATION SUMMARY"
    )

    print(
        "=" * 105
    )

    print(
        "Model:",
        args.model,
    )

    print(
        "Successful:",
        final_success,
        "/ 2000",
    )

    print(
        "Non-success rows:",
        final_errors,
    )

    print(
        "Generated successfully this invocation:",
        generated_this_run,
    )

    print(
        "Compact output:",
        paths[
            "compact"
        ],
    )

    print(
        "Journal:",
        paths[
            "journal"
        ],
    )

    if final_success != 2000:

        raise RuntimeError(
            f"Generation incomplete: "
            f"{final_success}/2000 successful."
        )


if __name__ == "__main__":
    main()
