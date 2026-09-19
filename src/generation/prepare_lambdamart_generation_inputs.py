#!/usr/bin/env python3

# Public path-portable copy of the frozen analysis source.
# Frozen analysis source SHA256: 84a5b2212a46b51f4c5db4924fcdbfe41d150c7215d4b0199e6de33764cea6e9

from pathlib import Path
import ast
import hashlib
import json
import re

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

LMOUT = (
    ROOT
    / "artifacts/jiis_lambdamart_reliability_20260918"
)

OUT = (
    ROOT
    / "artifacts/jiis_final_experiments_20260918"
)

CONTENT = (
    LMOUT
    / "content_only_lambdamart_scored_candidates.csv"
)

FULL = (
    LMOUT
    / "full_pacer_lambdamart_scored_candidates.csv"
)

CORPUS = (
    ROOT
    / "data/corpus/slide_corpus_final.jsonl"
)

FROZEN = (
    ROOT
    / "artifacts/final_matched_retrieval_freeze_v3"
    / "frozen_position_contexts_and_prompts_2000.jsonl"
)

ARCHIVED = (
    ROOT
    / "src/common/final_generation_reference.py"
)

EXPECTED_ARCHIVED_SHA = (
    "e672fc0c6d6bed46eb5760aeb5292d1bc7b29b64709b5a9db7ba8004f10cbbb6"
)

DEST = (
    OUT
    / "lambdamart_qwen3_generation_inputs_1200.jsonl"
)



OUT.mkdir(parents=True, exist_ok=True)

def sha256_file(path):
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


def sha256_text(s):
    return hashlib.sha256(
        str(s).encode(
            "utf-8"
        )
    ).hexdigest()


def read_jsonl(path):
    rows = []

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


def write_jsonl(path, rows):
    with path.open(
        "w",
        encoding="utf-8",
    ) as f:
        for r in rows:
            f.write(
                json.dumps(
                    r,
                    ensure_ascii=False,
                )
                + "\n"
            )


actual_sha = sha256_file(
    ARCHIVED
)

if actual_sha != EXPECTED_ARCHIVED_SHA:
    raise RuntimeError(
        "Archived source SHA mismatch. "
        f"expected={EXPECTED_ARCHIVED_SHA} "
        f"actual={actual_sha}"
    )

print(
    "Archived generation source SHA:",
    actual_sha,
)

# Recover the exact historical context formatter and prompt builder
# from the archived runtime source.
tree = ast.parse(
    ARCHIVED.read_text(
        encoding="utf-8"
    )
)

wanted = {
    "format_context",
    "make_prompt",
}

nodes = [
    n
    for n in tree.body
    if isinstance(
        n,
        ast.FunctionDef,
    )
    and n.name in wanted
]

if {
    n.name
    for n in nodes
} != wanted:
    raise RuntimeError(
        "Could not recover both historical helper functions."
    )

module = ast.Module(
    body=nodes,
    type_ignores=[],
)

ast.fix_missing_locations(
    module
)

ns = {
    "re": re,
    "MAX_CONTEXT_WORDS": 650,
}

exec(
    compile(
        module,
        str(ARCHIVED),
        "exec",
    ),
    ns,
    ns,
)

format_context = ns[
    "format_context"
]

make_prompt = ns[
    "make_prompt"
]


corpus = read_jsonl(
    CORPUS
)

for x in corpus:
    x["doc_id"] = str(
        x["doc_id"]
    )

corpus_by_doc = {
    x["doc_id"]: x
    for x in corpus
}

corpus_ids = set(
    corpus_by_doc
)


def detect_doc_column(df):
    aliases = [
        "doc",
        "doc_id",
        "candidate_doc_id",
        "document_id",
        "candidate_id",
        "slide_doc_id",
    ]

    valid = []

    for c in aliases:
        if c not in df.columns:
            continue

        vals = (
            df[c]
            .astype(str)
            .tolist()
        )

        frac = np.mean(
            [
                v in corpus_ids
                for v in vals
            ]
        )

        if frac == 1.0:
            valid.append(c)

    if len(valid) != 1:
        raise RuntimeError(
            "Could not uniquely identify document ID column. "
            f"valid={valid}, columns={list(df.columns)}"
        )

    return valid[0]


content = pd.read_csv(
    CONTENT
)

full = pd.read_csv(
    FULL
)

for x in [content, full]:
    x["task_id"] = (
        x["task_id"]
        .astype(str)
    )

doc_col_content = detect_doc_column(
    content
)

doc_col_full = detect_doc_column(
    full
)

if doc_col_content != doc_col_full:
    raise RuntimeError(
        "Content/full document-ID columns differ."
    )

DOC_COL = doc_col_content

print(
    "Candidate document column:",
    DOC_COL,
)


# Recover one canonical task record per task from the frozen,
# already-audited prior generation package.
frozen = read_jsonl(
    FROZEN
)

task_records = {}

for row in frozen:
    nested = row.get(
        "task",
        None,
    )

    task = (
        nested
        if isinstance(
            nested,
            dict,
        )
        else row
    )

    tid = (
        row.get("task_id")
        or task.get("task_id")
    )

    if tid is None:
        continue

    tid = str(tid)

    if tid not in task_records:
        task_records[tid] = {
            "row": row,
            "task": task,
        }


task_ids_content = set(
    content["task_id"].unique()
)

task_ids_full = set(
    full["task_id"].unique()
)

if (
    task_ids_content
    != task_ids_full
):
    raise RuntimeError(
        "Content/full task sets differ."
    )

if len(
    task_ids_content
) != 600:
    raise RuntimeError(
        f"Expected 600 answerable tasks, got {len(task_ids_content)}"
    )

missing_meta = (
    task_ids_content
    - set(task_records)
)

if missing_meta:
    raise RuntimeError(
        "Frozen metadata missing task IDs: "
        + repr(
            sorted(missing_meta)[:10]
        )
    )


def value_from(
    rec,
    names,
    default=None,
):
    row = rec["row"]
    task = rec["task"]

    for name in names:
        if name in task:
            v = task[name]

            if v is not None:
                return v

        if name in row:
            v = row[name]

            if v is not None:
                return v

    return default


def ranked_top3(frame):
    g = frame.copy()

    if "_input_order" not in g.columns:
        if "rrf_rank" in g.columns:
            g["_fallback_order"] = (
                g["rrf_rank"]
            )
        else:
            g["_fallback_order"] = (
                np.arange(len(g))
            )

        order_col = "_fallback_order"

    else:
        order_col = "_input_order"

    g = g.sort_values(
        ["score", order_col],
        ascending=[False, True],
        kind="mergesort",
    )

    return (
        g.iloc[:3][DOC_COL]
        .astype(str)
        .tolist()
    )


rows = []

conditions = [
    (
        "content_only_lambdamart",
        content,
    ),
    (
        "full_pacer_lambdamart",
        full,
    ),
]

for tid in sorted(
    task_ids_content
):
    meta = task_records[
        tid
    ]

    question = value_from(
        meta,
        [
            "question",
            "query",
        ],
    )

    reference = value_from(
        meta,
        [
            "reference_answer",
            "gold_answer",
            "expected_answer",
            "answer",
            "answer_text",
        ],
    )

    task_type = value_from(
        meta,
        [
            "task_type",
        ],
    )

    gold_docs = value_from(
        meta,
        [
            "evidence_doc_ids",
            "gold_evidence_doc_ids",
        ],
        default=[],
    )

    if question is None:
        raise RuntimeError(
            f"No question found for {tid}"
        )

    if reference is None:
        raise RuntimeError(
            f"No reference answer found for {tid}; "
            f"task keys={list(meta['task'].keys())}"
        )

    if not isinstance(
        gold_docs,
        list,
    ):
        gold_docs = (
            list(gold_docs)
            if gold_docs
            else []
        )

    target_lecture = int(
        content.loc[
            content["task_id"]
            == tid,
            "target_lecture",
        ].iloc[0]
    )

    for condition, frame in conditions:
        task_frame = frame[
            frame["task_id"]
            == tid
        ]

        raw_top3 = ranked_top3(
            task_frame
        )

        ctx, kept = format_context(
            raw_top3,
            corpus_by_doc,
        )

        prompt = make_prompt(
            "standard_rrf_bm25_tfidf_rag",
            str(question),
            ctx,
        )

        rows.append(
            {
                "run_id":
                    f"{condition}::{tid}",
                "condition":
                    condition,
                "task_id":
                    tid,
                "target_lecture":
                    target_lecture,
                "task_type":
                    task_type,
                "question":
                    str(question),
                "reference_answer":
                    str(reference),
                "gold_evidence_doc_ids":
                    [
                        str(x)
                        for x in gold_docs
                    ],
                "raw_top3_doc_ids":
                    raw_top3,
                "context_doc_ids":
                    [
                        str(x)
                        for x in kept
                    ],
                "context_text":
                    ctx,
                "prompt":
                    prompt,
                "prompt_sha256":
                    sha256_text(prompt),
                "historical_prompt_system_name":
                    "standard_rrf_bm25_tfidf_rag",
                "context_word_cap":
                    650,
                "raw_top_k":
                    3,
            }
        )


if len(rows) != 1200:
    raise RuntimeError(
        f"Expected 1200 generation inputs, got {len(rows)}"
    )

counts = (
    pd.DataFrame(rows)
    .groupby("condition")
    .size()
    .to_dict()
)

if counts != {
    "content_only_lambdamart": 600,
    "full_pacer_lambdamart": 600,
}:
    raise RuntimeError(
        f"Unexpected condition counts: {counts}"
    )


# Interleave paired conditions task-by-task.
rows = sorted(
    rows,
    key=lambda r: (
        r["task_id"],
        0
        if r["condition"]
        == "content_only_lambdamart"
        else 1,
    ),
)

write_jsonl(
    DEST,
    rows,
)

audit = (
    pd.DataFrame(
        [
            {
                "condition":
                    r["condition"],
                "task_id":
                    r["task_id"],
                "raw_top3_count":
                    len(
                        r[
                            "raw_top3_doc_ids"
                        ]
                    ),
                "delivered_count":
                    len(
                        r[
                            "context_doc_ids"
                        ]
                    ),
                "prompt_sha256":
                    r[
                        "prompt_sha256"
                    ],
            }
            for r in rows
        ]
    )
)

audit.to_csv(
    OUT
    / "lambdamart_qwen3_generation_input_audit.csv",
    index=False,
)

metadata = {
    "N_tasks": 600,
    "N_conditions": 2,
    "N_prompts": 1200,
    "conditions": [
        "content_only_lambdamart",
        "full_pacer_lambdamart",
    ],
    "generator":
        "Qwen/Qwen3-8B",
    "max_new_tokens":
        256,
    "do_sample":
        False,
    "raw_top_k":
        3,
    "context_word_cap":
        650,
    "prompt_builder":
        "verbatim archived make_prompt",
    "context_formatter":
        "verbatim archived format_context",
    "archived_source":
        str(ARCHIVED),
    "archived_source_sha256":
        actual_sha,
    "input_file":
        str(DEST),
    "input_sha256":
        sha256_file(DEST),
}

(
    OUT
    / "LAMBDAMART_QWEN3_INPUT_METADATA.json"
).write_text(
    json.dumps(
        metadata,
        indent=2,
    )
    + "\n",
    encoding="utf-8",
)

print("=" * 100)
print("LAMBDAMART QWEN3 INPUT FREEZE")
print("=" * 100)
print("Rows:", len(rows))
print("Counts:", counts)
print("Input:", DEST)
print("SHA256:", sha256_file(DEST))
print()
print("Delivered-context count distribution:")
print(
    audit.groupby(
        [
            "condition",
            "delivered_count",
        ]
    )
    .size()
    .to_string()
)
print()
print(
    "LAMBDAMART_QWEN3_INPUT_FREEZE = PASS"
)
