#!/usr/bin/env python3

# Public path-portable copy of the frozen analysis source.
# Frozen analysis source SHA256: 4524bd42bda8aa9d86c44608de85b84ca567807bb354cae0667c539b9366d78a

from pathlib import Path
import ast
from collections import Counter
import hashlib
import json
import re

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer


ROOT = Path(__file__).resolve().parents[2]

OUT = (
    ROOT
    / "artifacts/jiis_final_experiments_20260918"
)

GEN = (
    OUT
    / "qwen3_lambdamart_generation_1200_final.jsonl"
)

ARCHIVED = (
    ROOT
    / "src/common/final_generation_reference.py"
)

EXPECTED_ARCHIVED_SHA = (
    "e672fc0c6d6bed46eb5760aeb5292d1bc7b29b64709b5a9db7ba8004f10cbbb6"
)

SEM_MODEL = (
    "sentence-transformers/all-mpnet-base-v2"
)

SEED = 20260822
N_BOOT = 100000
N_PERM = 100000



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


if sha256(
    ARCHIVED
) != EXPECTED_ARCHIVED_SHA:
    raise RuntimeError(
        "Archived evaluator source SHA mismatch."
    )


# Recover exact historical lexical metric implementation.
tree = ast.parse(
    ARCHIVED.read_text(
        encoding="utf-8"
    )
)

wanted = {
    "norm",
    "metric_tokens",
    "lexical_f1",
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
        "Historical lexical metric helpers not found."
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
    "Counter": Counter,
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

lexical_f1 = ns[
    "lexical_f1"
]


rows = read_jsonl(
    GEN
)

if len(rows) != 1200:
    raise RuntimeError(
        f"Expected 1200 generations, got {len(rows)}"
    )

df = pd.DataFrame(rows)

if df["run_id"].nunique() != 1200:
    raise RuntimeError(
        "run_id uniqueness failed"
    )

if df["task_id"].nunique() != 600:
    raise RuntimeError(
        "task count failed"
    )

if df["error"].notna().any():
    raise RuntimeError(
        "Generation errors present"
    )

counts = (
    df.groupby("condition")
    .size()
    .to_dict()
)

expected_counts = {
    "content_only_lambdamart":
        600,
    "full_pacer_lambdamart":
        600,
}

if counts != expected_counts:
    raise RuntimeError(
        f"Condition counts wrong: {counts}"
    )


df[
    "lexical_f1"
] = [
    lexical_f1(
        str(a),
        str(r),
    )
    for a, r in zip(
        df["generated_answer"],
        df["reference_answer"],
    )
]


print("=" * 100)
print("LOADING SEMANTIC EVALUATOR")
print("=" * 100)
print(SEM_MODEL)

sem = SentenceTransformer(
    SEM_MODEL,
    device="cpu",
)

sem.max_seq_length = 384

answers = (
    df["generated_answer"]
    .astype(str)
    .tolist()
)

refs = (
    df["reference_answer"]
    .astype(str)
    .tolist()
)

answer_emb = sem.encode(
    answers,
    batch_size=32,
    convert_to_numpy=True,
    normalize_embeddings=True,
    show_progress_bar=True,
)

ref_emb = sem.encode(
    refs,
    batch_size=32,
    convert_to_numpy=True,
    normalize_embeddings=True,
    show_progress_bar=True,
)

df[
    "semantic_similarity"
] = (
    answer_emb
    * ref_emb
).sum(axis=1)


df[
    "answer_words"
] = (
    df["generated_answer"]
    .astype(str)
    .str.split()
    .str.len()
)

df.to_csv(
    OUT
    / "qwen3_lambdamart_generation_metrics_detail.csv",
    index=False,
)


wide = (
    df.pivot(
        index=[
            "task_id",
            "target_lecture",
            "task_type",
        ],
        columns="condition",
        values=[
            "lexical_f1",
            "semantic_similarity",
            "answer_words",
        ],
    )
)

wide.columns = [
    f"{metric}__{condition}"
    for metric, condition
    in wide.columns
]

wide = (
    wide.reset_index()
)


def cluster_ci_and_p(
    frame,
    diff_col,
    seed,
):
    agg = (
        frame.groupby(
            "target_lecture",
            as_index=False,
        )
        .agg(
            n=("task_id", "size"),
            s=(diff_col, "sum"),
        )
    )

    n = agg["n"].to_numpy(float)
    s = agg["s"].to_numpy(float)

    C = len(agg)
    rng = np.random.default_rng(seed)

    boot = np.empty(
        N_BOOT,
        dtype=float,
    )

    done = 0

    while done < N_BOOT:
        m = min(
            5000,
            N_BOOT - done,
        )

        ix = rng.integers(
            0,
            C,
            size=(m, C),
        )

        boot[
            done:done + m
        ] = (
            s[ix].sum(axis=1)
            / n[ix].sum(axis=1)
        )

        done += m

    lo, hi = np.quantile(
        boot,
        [0.025, 0.975],
    )

    observed = (
        s.sum()
        / n.sum()
    )

    # Lecture-cluster random sign-flip test.
    perm = np.empty(
        N_PERM,
        dtype=float,
    )

    done = 0

    while done < N_PERM:
        m = min(
            5000,
            N_PERM - done,
        )

        signs = rng.choice(
            [-1.0, 1.0],
            size=(m, C),
        )

        perm[
            done:done + m
        ] = (
            (
                signs
                * s[None, :]
            ).sum(axis=1)
            / n.sum()
        )

        done += m

    p = (
        1
        + int(
            (
                np.abs(perm)
                >= abs(observed)
            ).sum()
        )
    ) / (
        N_PERM + 1
    )

    lecture_mean_diff = (
        frame.groupby(
            "target_lecture"
        )[diff_col]
        .mean()
    )

    improved = int(
        (
            lecture_mean_diff
            > 1e-15
        ).sum()
    )

    tied = int(
        np.isclose(
            lecture_mean_diff,
            0.0,
            atol=1e-15,
        ).sum()
    )

    worse = int(
        (
            lecture_mean_diff
            < -1e-15
        ).sum()
    )

    return (
        float(lo),
        float(hi),
        float(p),
        improved,
        tied,
        worse,
    )


summary_rows = []

metrics = [
    "lexical_f1",
    "semantic_similarity",
]

for idx, metric in enumerate(metrics):
    c = (
        f"{metric}"
        "__content_only_lambdamart"
    )

    f = (
        f"{metric}"
        "__full_pacer_lambdamart"
    )

    diff = (
        metric
        + "_full_minus_content"
    )

    wide[diff] = (
        wide[f]
        - wide[c]
    )

    (
        lo,
        hi,
        p,
        improved,
        tied,
        worse,
    ) = cluster_ci_and_p(
        wide,
        diff,
        SEED + idx,
    )

    summary_rows.append(
        {
            "metric":
                metric,
            "N":
                len(wide),
            "n_lectures":
                int(
                    wide[
                        "target_lecture"
                    ].nunique()
                ),
            "content_only_mean":
                float(
                    wide[c].mean()
                ),
            "full_pacer_mean":
                float(
                    wide[f].mean()
                ),
            "delta_full_minus_content":
                float(
                    wide[diff].mean()
                ),
            "cluster_bootstrap_ci95_low":
                lo,
            "cluster_bootstrap_ci95_high":
                hi,
            "cluster_signflip_p_two_sided":
                p,
            "lectures_improved":
                improved,
            "lectures_tied":
                tied,
            "lectures_worse":
                worse,
        }
    )


summary = pd.DataFrame(
    summary_rows
)

summary.to_csv(
    OUT
    / "qwen3_lambdamart_generation_primary_summary.csv",
    index=False,
)


# Descriptive task-type breakdown.
task_rows = []

for task_type, g in wide.groupby(
    "task_type",
    dropna=False,
):
    for metric in metrics:
        c = (
            f"{metric}"
            "__content_only_lambdamart"
        )

        f = (
            f"{metric}"
            "__full_pacer_lambdamart"
        )

        task_rows.append(
            {
                "task_type":
                    task_type,
                "metric":
                    metric,
                "N":
                    len(g),
                "content_only_mean":
                    float(g[c].mean()),
                "full_pacer_mean":
                    float(g[f].mean()),
                "delta_full_minus_content":
                    float(
                        (
                            g[f]
                            - g[c]
                        ).mean()
                    ),
            }
        )

pd.DataFrame(
    task_rows
).to_csv(
    OUT
    / "qwen3_lambdamart_generation_by_task_type.csv",
    index=False,
)

wide.to_csv(
    OUT
    / "qwen3_lambdamart_generation_paired_task_detail.csv",
    index=False,
)


metadata = {
    "generator":
        "Qwen/Qwen3-8B",
    "conditions": [
        "Content-only LambdaMART top-3",
        "Full PACER LambdaMART top-3",
    ],
    "N_tasks":
        600,
    "N_generations":
        1200,
    "lexical_metric":
        "verbatim archived lexical_f1",
    "semantic_model":
        SEM_MODEL,
    "semantic_max_seq_length":
        384,
    "cluster_unit":
        "target lecture",
    "bootstrap_replicates":
        N_BOOT,
    "signflip_replicates":
        N_PERM,
    "generation_file_sha256":
        sha256(GEN),
}

(
    OUT
    / "QWEN3_LAMBDAMART_EVAL_METADATA.json"
).write_text(
    json.dumps(
        metadata,
        indent=2,
    )
    + "\n",
    encoding="utf-8",
)

(
    OUT
    / "QWEN3_LAMBDAMART_EVALUATION_COMPLETE.txt"
).write_text(
    "PASS\n",
    encoding="utf-8",
)

print()
print("=" * 100)
print("QWEN3 PAIRED END-TO-END RESULT")
print("=" * 100)
print(
    summary.to_string(
        index=False
    )
)

print()
print("=" * 100)
print("DESCRIPTIVE TASK-TYPE BREAKDOWN")
print("=" * 100)
print(
    pd.DataFrame(
        task_rows
    ).to_string(
        index=False
    )
)

print()
print(
    "QWEN3_LAMBDAMART_EVALUATION_COMPLETE = PASS"
)
