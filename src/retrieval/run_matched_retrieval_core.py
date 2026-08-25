#!/usr/bin/env python3

from pathlib import Path
from collections import defaultdict, Counter
import ast
import hashlib
import json
import math
import re

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

ROOT = Path(__file__).resolve().parents[2]

PIPE = ROOT / "src/common/runtime_reference.py"


CORPUS = ROOT / "data/corpus/slide_corpus_final.jsonl"
BENCH = ROOT / "data/benchmark/core_qa_benchmark.jsonl"

QWEN3_FINAL = (
    ROOT
    / "artifacts/reference_generation"
    / "qwen3_8b_full1000_all4systems_4000.jsonl"
)


OLD_POS = (
    ROOT
    / "artifacts/position_constrained_retrieval"
    / "position_constrained_retrieval_detail.csv"
)

OUT = ROOT / "artifacts/matched_retrieval"
OUT.mkdir(parents=True, exist_ok=True)

TOP_K = 3
COMPONENT_TOP_K = 50
K_SWEEP = [1, 3, 5, 10, 20, 50]

ANSWERABLE = {
    "evidence_cited_qa",
    "neighbor_slide_conceptual_qa",
    "slide_local_factual_qa",
}


# ============================================================================
# Utilities
# ============================================================================

def read_jsonl(path):
    return [
        json.loads(line)
        for line in path.open("r", encoding="utf-8")
        if line.strip()
    ]


def sha256_text(x):
    return hashlib.sha256(
        str(x).encode("utf-8")
    ).hexdigest()


def function_source(text, tree, name):
    lines = text.splitlines()

    for node in tree.body:
        if (
            isinstance(node, ast.FunctionDef)
            and node.name == name
        ):
            return "\n".join(
                lines[node.lineno - 1:node.end_lineno]
            )

    raise RuntimeError(
        f"Function not found: {name}"
    )


def recover_helpers(path):
    text = path.read_text(
        encoding="utf-8",
        errors="replace",
    )

    tree = ast.parse(text)

    wanted = {
        "norm",
        "tokenize",
        "unique",
        "doc_text",
        "format_context",
        "simple_bm25_scores",
        "rrf",
        "make_prompt",
    }

    body = []

    for node in tree.body:

        if isinstance(node, ast.Assign):
            try:
                ast.literal_eval(node.value)
            except Exception:
                continue
            body.append(node)

        elif isinstance(node, ast.AnnAssign):
            try:
                ast.literal_eval(node.value)
            except Exception:
                continue
            body.append(node)

    for node in tree.body:
        if (
            isinstance(node, ast.FunctionDef)
            and node.name in wanted
        ):
            body.append(node)

    mod = ast.Module(
        body=body,
        type_ignores=[],
    )

    ast.fix_missing_locations(mod)

    ns = {
        "re": re,
        "json": json,
        "np": np,
        "math": math,
        "defaultdict": defaultdict,
    }

    exec(
        compile(mod, str(path), "exec"),
        ns,
        ns,
    )

    for name in [
        "tokenize",
        "doc_text",
        "format_context",
        "simple_bm25_scores",
        "rrf",
        "make_prompt",
    ]:
        if name not in ns:
            raise RuntimeError(
                f"Could not recover {name}"
            )

    return ns, text, tree


def pos_value(doc):
    p = doc.get("course_position")

    if p is None or str(p) == "":
        raise RuntimeError(
            f"Missing course_position for {doc.get('doc_id')}"
        )

    return int(p)


def first_rank(ranking, relevant):
    rel = set(relevant)

    for i, d in enumerate(ranking, start=1):
        if d in rel:
            return i

    return None


# ============================================================================
# Required files
# ============================================================================

for p in [
    PIPE,
    CORPUS,
    BENCH,
    QWEN3_FINAL,
]:
    if not p.exists():
        raise SystemExit(
            f"FAILED: missing {p}"
        )


# ============================================================================
# Data
# ============================================================================

corpus = read_jsonl(CORPUS)
tasks = read_jsonl(BENCH)
saved_outputs = read_jsonl(QWEN3_FINAL)

if len(corpus) != 1117:
    raise SystemExit(
        f"FAILED: expected 1117 corpus docs, got {len(corpus)}"
    )

if len(tasks) != 1000:
    raise SystemExit(
        f"FAILED: expected 1000 tasks, got {len(tasks)}"
    )

if len(saved_outputs) != 4000:
    raise SystemExit(
        f"FAILED: expected 4000 saved final outputs, got {len(saved_outputs)}"
    )

corpus_by_doc = {
    str(x["doc_id"]): x
    for x in corpus
}

doc_ids = [
    str(x["doc_id"])
    for x in corpus
]

docs = []

ns, source_text, source_tree = recover_helpers(
    PIPE
)

tokenize = ns["tokenize"]
doc_text = ns["doc_text"]
format_context = ns["format_context"]
simple_bm25_scores = ns["simple_bm25_scores"]
rrf = ns["rrf"]
make_prompt = ns["make_prompt"]

for x in corpus:
    docs.append(
        doc_text(x)
    )

doc_tokens = [
    tokenize(x)
    for x in docs
]

print("=" * 110)
print("MATCHED RETRIEVAL")
print("=" * 110)

print("Corpus:", len(corpus))
print("Tasks:", len(tasks))
print("TOP_K:", TOP_K)
print("COMPONENT_TOP_K:", COMPONENT_TOP_K)
print("MAX_CONTEXT_WORDS:", ns.get("MAX_CONTEXT_WORDS"))

print()
print("Reference helper hashes:")

for fn in [
    "tokenize",
    "doc_text",
    "simple_bm25_scores",
    "rrf",
    "format_context",
    "make_prompt",
]:
    print(
        fn,
        sha256_text(
            function_source(
                source_text,
                source_tree,
                fn,
            )
        ),
    )


# ============================================================================
# Saved original context IDs
# ============================================================================

saved = {}

for r in saved_outputs:

    system = str(r["system_name"])

    if system not in {
        "standard_bm25_rag",
        "standard_rrf_bm25_tfidf_rag",
    }:
        continue

    saved[
        (
            str(r["task_id"]),
            system,
        )
    ] = [
        str(x)
        for x in r.get(
            "context_doc_ids",
            []
        )
    ]

if len(saved) != 2000:
    raise SystemExit(
        f"FAILED: expected 2000 saved BM25/RRF contexts, got {len(saved)}"
    )


# ============================================================================
# A. Exact ORIGINAL global retrieval reconstruction
#
# IMPORTANT:
# This literally mirrors reference source lines 272-290:
#
#   tfidf = TfidfVectorizer(...)
#   X = tfidf.fit_transform(docs)
#   Q = tfidf.transform(questions)
#   tfidf_scores = Q @ X.T
#
#   bm_scores = simple_bm25_scores(...)
#   bm_order = np.argsort(bm_scores)[::-1][:50]
#   bm25_rank = [...]
#
#   tf_order = np.argsort(tf_row)[::-1][:50]
#   tfidf_rank = [...]
#
#   rrf_rank = rrf([bm25_rank, tfidf_rank])[:50]
#
# Then top-3 goes through format_context().
# ============================================================================

questions = [
    t["question"]
    for t in tasks
]

tfidf_global = TfidfVectorizer(
    lowercase=True,
    stop_words="english",
    ngram_range=(1, 3),
    max_df=0.85,
)

X_global = tfidf_global.fit_transform(
    docs
)

Q_global = tfidf_global.transform(
    questions
)

TF_GLOBAL = Q_global @ X_global.T

global_reconstruction = []

for i, task in enumerate(tasks):

    tid = str(task["task_id"])

    bm_scores = simple_bm25_scores(
        tokenize(
            task["question"]
        ),
        doc_tokens,
    )

    bm_order = np.argsort(
        bm_scores
    )[::-1][:COMPONENT_TOP_K]

    bm25_rank = [
        doc_ids[j]
        for j in bm_order
    ]

    tf_row = (
        TF_GLOBAL[i]
        .toarray()
        .ravel()
    )

    tf_order = np.argsort(
        tf_row
    )[::-1][:COMPONENT_TOP_K]

    tfidf_rank = [
        doc_ids[j]
        for j in tf_order
    ]

    rrf_rank = rrf(
        [
            bm25_rank,
            tfidf_rank,
        ]
    )[:COMPONENT_TOP_K]

    for method, system, rank in [
        (
            "bm25",
            "standard_bm25_rag",
            bm25_rank,
        ),
        (
            "rrf",
            "standard_rrf_bm25_tfidf_rag",
            rrf_rank,
        ),
    ]:

        ctx, kept = format_context(
            rank[:TOP_K],
            corpus_by_doc,
        )

        kept = [
            str(x)
            for x in kept
        ]

        gold_saved = saved[
            (
                tid,
                system,
            )
        ]

        global_reconstruction.append(
            {
                "task_id":
                    tid,

                "method":
                    method,

                "raw_top3":
                    json.dumps(
                        rank[:TOP_K]
                    ),

                "formatter_kept_ids":
                    json.dumps(
                        kept
                    ),

                "saved_final_ids":
                    json.dumps(
                        gold_saved
                    ),

                "exact":
                    int(
                        kept
                        == gold_saved
                    ),
            }
        )


recon_df = pd.DataFrame(
    global_reconstruction
)

recon_summary = (
    recon_df
    .groupby("method")
    .agg(
        N=("task_id", "size"),
        ExactCount=("exact", "sum"),
        ExactRate=("exact", "mean"),
    )
    .reset_index()
)

print()
print("=" * 110)
print("EXACT ORIGINAL GLOBAL RECONSTRUCTION")
print("=" * 110)

print(
    recon_summary.to_string(
        index=False
    )
)

GLOBAL_EXACT = bool(
    (
        recon_summary[
            "ExactCount"
        ]
        == 1000
    ).all()
)

print()
print(
    "GLOBAL_RETRIEVAL_EXACTLY_RECONSTRUCTED =",
    GLOBAL_EXACT,
)

recon_df.to_csv(
    OUT
    / "exact_original_global_reconstruction.csv",
    index=False,
)

if not GLOBAL_EXACT:

    failures = recon_df.loc[
        recon_df["exact"].eq(0)
    ]

    failures.to_csv(
        OUT
        / "global_reconstruction_failures.csv",
        index=False,
    )

    print()
    print(
        "STOP: exact historical global retrieval did not reconstruct."
    )

    print(
        failures.head(30).to_string(
            index=False
        )
    )

    raise SystemExit(2)


# ============================================================================
# B. Final POSITION-CONSTRAINED retrieval
#
# Experimental intervention:
#
#   ONLY course-position availability changes.
#
# We deliberately HOLD FIXED:
#   - original 1,117-slide corpus statistics
#   - reference BM25 scorer
#   - globally fitted reference TF-IDF representation
#   - exact np.argsort(...)[::-1] ranking semantics
#   - component top-50 truncation
#   - reference RRF implementation
#
# For each task:
#
#   1. Compute the exact historical full-corpus BM25 ranking.
#   2. Compute the exact historical full-corpus TF-IDF ranking.
#   3. Remove documents whose course_position is AFTER the target.
#   4. From the remaining eligible ranking, take the first 50 BM25 docs
#      and first 50 TF-IDF docs.
#   5. Apply the exact reference RRF to those two top-50 lists.
#   6. Take the first 3 documents and pass them through the exact
#      reference format_context() function.
#
# This is a filtered-index / eligibility-mask experiment. It does NOT
# refit TF-IDF or recompute BM25 corpus statistics for every prefix.
# Therefore course availability is the only retrieval intervention.
# ============================================================================

position_rows = []
rank_store = {}

for ti, task in enumerate(tasks):

    tid = str(
        task["task_id"]
    )

    target = str(
        task["target_doc_id"]
    )

    if target not in corpus_by_doc:
        raise RuntimeError(
            f"Target missing from corpus: {tid} {target}"
        )

    target_position = pos_value(
        corpus_by_doc[target]
    )

    eligible_set = {
        doc_ids[j]
        for j, d in enumerate(corpus)
        if pos_value(d) <= target_position
    }

    # --------------------------------------------------------
    # BM25
    #
    # Exact historical FULL-CORPUS scoring.
    # Difference from original condition:
    # future documents are filtered before top-50 truncation.
    # --------------------------------------------------------

    bm_scores = simple_bm25_scores(
        tokenize(
            task["question"]
        ),
        doc_tokens,
    )

    bm_order_full = np.argsort(
        bm_scores
    )[::-1]

    bm_rank_eligible_full = [
        doc_ids[j]
        for j in bm_order_full
        if doc_ids[j] in eligible_set
    ]

    bm_rank50 = (
        bm_rank_eligible_full[
            :COMPONENT_TOP_K
        ]
    )

    # --------------------------------------------------------
    # TF-IDF
    #
    # Exact historical globally fitted TF-IDF score row.
    # No prefix-specific refitting.
    # --------------------------------------------------------

    tf_row = (
        TF_GLOBAL[ti]
        .toarray()
        .ravel()
    )

    tf_order_full = np.argsort(
        tf_row
    )[::-1]

    tf_rank_eligible_full = [
        doc_ids[j]
        for j in tf_order_full
        if doc_ids[j] in eligible_set
    ]

    tf_rank50 = (
        tf_rank_eligible_full[
            :COMPONENT_TOP_K
        ]
    )

    # --------------------------------------------------------
    # RRF
    #
    # Exact reference structure:
    # top-50 BM25 + top-50 TF-IDF -> rrf() -> top-50.
    # --------------------------------------------------------

    rrf_rank = rrf(
        [
            bm_rank50,
            tf_rank50,
        ]
    )[:COMPONENT_TOP_K]

    # BM25 has a complete eligible ordering.
    rank_store[
        (
            tid,
            "bm25",
        )
    ] = bm_rank_eligible_full

    # Historical RRF is explicitly constructed only through rank 50.
    rank_store[
        (
            tid,
            "rrf",
        )
    ] = rrf_rank

    for method, ranking in [
        (
            "bm25",
            bm_rank50,
        ),
        (
            "rrf",
            rrf_rank,
        ),
    ]:

        context_text, kept = format_context(
            ranking[:TOP_K],
            corpus_by_doc,
        )

        kept = [
            str(x)
            for x in kept
        ]

        position_rows.append(
            {
                "task_id":
                    tid,

                "task_type":
                    task["task_type"],

                "retrieval_method":
                    method,

                "system_name":
                    (
                        f"{method}_position_constrained_rag_matched"
                    ),

                "original_prompt_system_name":
                    (
                        "standard_bm25_rag"
                        if method == "bm25"
                        else "standard_rrf_bm25_tfidf_rag"
                    ),

                "target_doc_id":
                    target,

                "target_course_position":
                    target_position,

                "candidate_count":
                    len(eligible_set),

                "retrieved_top3_raw":
                    json.dumps(
                        ranking[:TOP_K]
                    ),

                "context_doc_ids":
                    json.dumps(
                        kept
                    ),

                "context_text":
                    context_text,
            }
        )


position_df = pd.DataFrame(
    position_rows
)

if len(position_df) != 2000:
    raise SystemExit(
        f"FAILED: expected 2000 position rows; got {len(position_df)}"
    )

if position_df.duplicated(
    subset=[
        "task_id",
        "retrieval_method",
    ]
).any():
    raise SystemExit(
        "FAILED: duplicate position task/method pairs."
    )


# ============================================================================
# C. Attach exact reference prompts
# ============================================================================

task_by_id = {
    str(x["task_id"]): x
    for x in tasks
}

frozen_rows = []

for _, r in position_df.iterrows():

    task = task_by_id[
        str(
            r["task_id"]
        )
    ]

    original_system = str(
        r[
            "original_prompt_system_name"
        ]
    )

    prompt = make_prompt(
        original_system,
        task["question"],
        r["context_text"],
    )

    frozen_rows.append(
        {
            "task_id":
                r["task_id"],

            "task_type":
                r["task_type"],

            "retrieval_method":
                r[
                    "retrieval_method"
                ],

            "system_name":
                r[
                    "system_name"
                ],

            "original_prompt_system_name":
                original_system,

            "target_doc_id":
                task[
                    "target_doc_id"
                ],

            "gold_evidence_doc_ids":
                json.dumps(
                    task.get(
                        "evidence_doc_ids",
                        []
                    )
                ),

            "question":
                task[
                    "question"
                ],

            "reference_answer":
                task[
                    "reference_answer"
                ],

            "candidate_count":
                r[
                    "candidate_count"
                ],

            "retrieved_top3_raw":
                r[
                    "retrieved_top3_raw"
                ],

            "context_doc_ids":
                r[
                    "context_doc_ids"
                ],

            "context_text":
                r[
                    "context_text"
                ],

            "prompt":
                prompt,

            "prompt_sha256":
                sha256_text(
                    prompt
                ),
        }
    )


frozen = pd.DataFrame(
    frozen_rows
)

if len(frozen) != 2000:
    raise SystemExit(
        f"FAILED: expected 2000 frozen prompts, got {len(frozen)}"
    )

if frozen.duplicated(
    subset=[
        "task_id",
        "retrieval_method",
    ]
).any():
    raise SystemExit(
        "FAILED: duplicate task/method pairs."
    )

frozen.to_csv(
    OUT
    / "frozen_position_contexts_and_prompts_2000.csv",
    index=False,
)

with (
    OUT
    / "frozen_position_contexts_and_prompts_2000.jsonl"
).open(
    "w",
    encoding="utf-8",
) as f:

    for row in frozen.to_dict(
        "records"
    ):
        f.write(
            json.dumps(
                row,
                ensure_ascii=False,
            )
            + "\n"
        )


# ============================================================================
# D. Position-constrained retrieval metrics
# ============================================================================

metric_rows = []
mrr_rows = []

for task in tasks:

    tid = str(task["task_id"])
    task_type = task["task_type"]

    target = str(
        task["target_doc_id"]
    )

    gold = [
        str(x)
        for x in task.get(
            "evidence_doc_ids",
            []
        )
    ]

    gold_set = set(gold)

    target_lecture = str(
        corpus_by_doc[
            target
        ][
            "lecture_id"
        ]
    )

    for method in [
        "bm25",
        "rrf",
    ]:

        ranking = rank_store[
            (
                tid,
                method,
            )
        ]

        first_gold = first_rank(
            ranking,
            gold,
        )

        first_target = first_rank(
            ranking,
            [
                target
            ],
        )

        mrr_rows.append(
            {
                "task_id":
                    tid,

                "task_type":
                    task_type,

                "method":
                    method,

                "first_gold_rank":
                    first_gold,

                "first_target_rank":
                    first_target,

                "rr_any_gold":
                    (
                        1.0 / first_gold
                        if first_gold
                        else 0.0
                    ),

                "rr_target":
                    (
                        1.0 / first_target
                        if first_target
                        else 0.0
                    ),
            }
        )

        for k in K_SWEEP:

            top = ranking[:k]
            top_set = set(top)

            any_gold = int(
                bool(
                    top_set
                    & gold_set
                )
            )

            all_gold = int(
                bool(gold_set)
                and gold_set.issubset(
                    top_set
                )
            )

            target_hit = int(
                target in top_set
            )

            lecture_hit = int(
                any(
                    str(
                        corpus_by_doc[d][
                            "lecture_id"
                        ]
                    )
                    == target_lecture
                    for d in top
                )
            )

            metric_rows.append(
                {
                    "task_id":
                        tid,

                    "task_type":
                        task_type,

                    "method":
                        method,

                    "k":
                        k,

                    "any_gold":
                        any_gold,

                    "all_gold":
                        all_gold,

                    "target_slide":
                        target_hit,

                    "target_lecture":
                        lecture_hit,
                }
            )


metric_df = pd.DataFrame(
    metric_rows
)

mrr_df = pd.DataFrame(
    mrr_rows
)

metric_df.to_csv(
    OUT
    / "position_retrieval_k_sweep_detail.csv",
    index=False,
)

mrr_df.to_csv(
    OUT
    / "position_retrieval_mrr_detail.csv",
    index=False,
)

answer_metric = metric_df.loc[
    metric_df[
        "task_type"
    ].isin(
        ANSWERABLE
    )
]

answer_mrr = mrr_df.loc[
    mrr_df[
        "task_type"
    ].isin(
        ANSWERABLE
    )
]

k_summary = (
    answer_metric
    .groupby(
        [
            "method",
            "k",
        ]
    )
    .agg(
        N=("task_id", "size"),
        AnyGoldRecall=("any_gold", "mean"),
        AllGoldRecall=("all_gold", "mean"),
        TargetSlideRecall=("target_slide", "mean"),
        TargetLectureRecall=("target_lecture", "mean"),
    )
    .reset_index()
)

mrr_summary = (
    answer_mrr
    .groupby(
        "method"
    )
    .agg(
        N=("task_id", "size"),
        MRRAnyGold=("rr_any_gold", "mean"),
        MRRTargetSlide=("rr_target", "mean"),
        MedianFirstGoldRank=("first_gold_rank", "median"),
        MedianFirstTargetRank=("first_target_rank", "median"),
    )
    .reset_index()
)

k_summary.to_csv(
    OUT
    / "position_retrieval_k_sweep_answerable.csv",
    index=False,
)

mrr_summary.to_csv(
    OUT
    / "position_retrieval_mrr_answerable.csv",
    index=False,
)


# ============================================================================
# E. Compare with previous 1112-slide position run
# ============================================================================

comparison_rows = []

if OLD_POS.exists():

    old = pd.read_csv(
        OLD_POS
    )

    for method in [
        "bm25",
        "rrf",
    ]:

        old_sub = old.loc[
            old["method"]
            .astype(str)
            .eq(
                f"{method}_position_constrained"
            )
        ]

        new_sub = frozen.loc[
            frozen[
                "retrieval_method"
            ].eq(
                method
            )
        ]

        old_map = {
            str(r["task_id"]):
                json.loads(
                    r[
                        "retrieved_ids"
                    ]
                )
            for _, r
            in old_sub.iterrows()
        }

        new_map = {
            str(r["task_id"]):
                json.loads(
                    r[
                        "retrieved_top3_raw"
                    ]
                )
            for _, r
            in new_sub.iterrows()
        }

        shared = sorted(
            set(old_map)
            & set(new_map)
        )

        exact = sum(
            old_map[t]
            == new_map[t]
            for t in shared
        )

        comparison_rows.append(
            {
                "method":
                    method,

                "shared_tasks":
                    len(shared),

                "exact_raw_top3":
                    exact,

                "exact_rate":
                    (
                        exact
                        / len(shared)
                        if shared
                        else np.nan
                    ),
            }
        )


comparison_df = pd.DataFrame(
    comparison_rows
)

comparison_df.to_csv(
    OUT
    / "comparison_with_previous_1112_position_run.csv",
    index=False,
)


# ============================================================================
# F. Freeze metadata
# ============================================================================

metadata = {
    "status":
        (
            "FINAL MATCHED POSITION RETRIEVAL/PROMPTS FROZEN; "
            "NO GENERATION RUN"
        ),

    "benchmark":
        str(BENCH),

    "benchmark_tasks":
        1000,

    "corpus":
        str(CORPUS),

    "corpus_docs":
        1117,

    "reference_pipeline":
        str(PIPE),

    "historical_global_reconstruction":
        {
            "bm25_exact":
                int(
                    recon_summary.loc[
                        recon_summary[
                            "method"
                        ].eq(
                            "bm25"
                        ),
                        "ExactCount",
                    ].iloc[0]
                ),

            "rrf_exact":
                int(
                    recon_summary.loc[
                        recon_summary[
                            "method"
                        ].eq(
                            "rrf"
                        ),
                        "ExactCount",
                    ].iloc[0]
                ),
        },

    "position_constraint_definition":
        (
            "Hold the exact reference full-corpus BM25/TF-IDF scoring fixed; "
            "filter documents after the target course_position before "
            "the historical top-50 component truncation; then apply the "
            "same reference RRF implementation."
        ),

    "bm25":
        {
            "scorer":
                "reference simple_bm25_scores",

            "score_scope":
                "original full 1,117-slide corpus; eligibility filter applied after scoring",

            "sort":
                "np.argsort(scores)[::-1]",

            "component_top_k":
                50,
        },

    "tfidf":
        {
            "vectorizer":
                "sklearn TfidfVectorizer",

            "fit_scope":
                "original full 1,117-slide corpus; eligibility filter applied after scoring",

            "lowercase":
                True,

            "stop_words":
                "english",

            "ngram_range":
                [1, 3],

            "max_df":
                0.85,

            "sort":
                "np.argsort(scores)[::-1]",

            "component_top_k":
                50,
        },

    "rrf":
        {
            "function":
                "reference rrf",

            "k":
                60,

            "inputs":
                "top-50 BM25 + top-50 TF-IDF",

            "output_truncation":
                50,
        },

    "generation_context_top_k":
        3,

    "context_formatter":
        "exact reference format_context, MAX_CONTEXT_WORDS=650",

    "prompt_builder":
        "exact reference make_prompt",

    "planned_generation":
        {
            "models":
                3,

            "retrieval_conditions":
                2,

            "tasks":
                1000,

            "new_generation_requests":
                6000,

            "original_evaluated_outputs":
                15000,

            "total_evaluated_if_completed":
                21000,
        },

    "generation_decoding":
        {
            "max_new_tokens":
                256,

            "do_sample":
                False,
        },

    "bge":
        (
            "Retained as retrieval-only baseline; excluded from matched "
            "end-to-end generation because no historical BGE generation "
            "condition existed."
        ),
}

(
    OUT
    / "FINAL_MATCHED_POSITION_PROTOCOL_FREEZE.json"
).write_text(
    json.dumps(
        metadata,
        indent=2,
    ),
    encoding="utf-8",
)


# ============================================================================
# Report
# ============================================================================

report = []

report.append(
    "=" * 110
)

report.append(
    "MATCHED POSITION RETRIEVAL"
)

report.append(
    "=" * 110
)

report.append("")

report.append(
    "Historical global reconstruction:"
)

report.append(
    recon_summary.to_string(
        index=False
    )
)

report.append("")

report.append(
    f"GLOBAL_RETRIEVAL_EXACTLY_RECONSTRUCTED = {GLOBAL_EXACT}"
)

report.append("")

report.append(
    "Final candidate corpus: 1,117 original slides."
)

report.append(
    "Position constraint: preserve original full-corpus scoring/index; filter future documents before top-50 truncation."
)

report.append(
    "Retrieval implementation: exact reference BM25 / TF-IDF / top-50 RRF logic with a course-position eligibility mask."
)

report.append(
    "Context: exact reference top-3 formatter with 650-word cap."
)

report.append(
    "Prompt: exact reference system-specific prompt builder."
)

report.append("")

report.append(
    "Answerable-task position-constrained retrieval k-sweep:"
)

report.append(
    k_summary.to_string(
        index=False
    )
)

report.append("")

report.append(
    "Answerable-task position-constrained MRR:"
)

report.append(
    mrr_summary.to_string(
        index=False
    )
)

report.append("")

report.append(
    "Comparison with prior 1,112-slide position run:"
)

if len(comparison_df):
    report.append(
        comparison_df.to_string(
            index=False
        )
    )
else:
    report.append(
        "Prior position run unavailable."
    )

report.append("")

report.append(
    f"Frozen unique position prompts: {len(frozen)}"
)

report.append(
    "Replicated over three models: 6,000 matched generation requests."
)

report.append("")

report.append(
    "No generation was performed."
)

report_text = "\n".join(
    report
)

(
    OUT
    / "FINAL_MATCHED_POSITION_FREEZE_REPORT.txt"
).write_text(
    report_text + "\n",
    encoding="utf-8",
)

print()
print(report_text)

print()
print("=" * 110)
print("DONE — NO GENERATION")
print("=" * 110)
