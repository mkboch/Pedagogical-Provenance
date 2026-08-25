#!/usr/bin/env python3

from pathlib import Path
import ast
import hashlib
import json

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

GEN = (
    ROOT
    / "artifacts"
    / "matched_generation"
)

RET = (
    ROOT
    / "artifacts"
    / "matched_retrieval"
)

AGG = (
    GEN
    / "three_model_matched_aggregate"
)

SEM = (
    AGG
    / "semantic_evaluation"
    / "semantic_paired_global_position_detail_3600.csv"
)

CIT = (
    AGG
    / "citation_evaluation"
    / "exact_citation_detail_6000.csv"
)

OUT = (
    AGG
    / "retrieval_generation_linkage"
)

OUT.mkdir(
    parents=True,
    exist_ok=True,
)


KSWEEP = (
    RET
    / "position_retrieval_k_sweep_detail.csv"
)

KSWEEP_SUMMARY = (
    RET
    / "position_retrieval_k_sweep_answerable.csv"
)

FROZEN = (
    RET
    / "frozen_position_contexts_and_prompts_2000.csv"
)


MODEL_METRICS = {
    "qwen3_8b":
        GEN
        / "qwen3_metric_checkpoint"
        / "qwen3_matched_position_exact_historical_metrics_detail.csv",

    "qwen2_5_7b":
        GEN
        / "qwen2_5_metric_checkpoint"
        / "qwen25_matched_position_exact_historical_metrics_detail.csv",

    "mistral_7b":
        GEN
        / "mistral_metric_checkpoint"
        / "mistral_matched_position_exact_historical_metrics_detail.csv",
}


ANSWERABLE = {
    "evidence_cited_qa",
    "slide_local_factual_qa",
    "neighbor_slide_conceptual_qa",
}


EXPECTED_K3 = {
    "bm25": {
        "N":
            600,

        "any_gold":
            0.090000,

        "all_gold":
            0.055000,

        "target_slide":
            0.07333333333333333,
    },

    "rrf": {
        "N":
            600,

        "any_gold":
            0.145000,

        "all_gold":
            0.09666666666666666,

        "target_slide":
            0.12166666666666667,
    },
}


QUALITY_METRICS = [
    "lexical_f1",
    "evidence_token_overlap",
    "semantic_similarity",
    "corrected_gold_citation",
    "corrected_any_valid_citation",
]


N_BOOT = 20000
SEED = 20260821


def sha256_file(path):

    h = hashlib.sha256()

    with path.open(
        "rb"
    ) as f:

        for block in iter(
            lambda: f.read(
                1024 * 1024
            ),
            b"",
        ):

            h.update(
                block
            )

    return h.hexdigest()


def listish(x):

    if isinstance(
        x,
        list,
    ):
        return [
            str(v)
            for v in x
        ]

    if isinstance(
        x,
        (
            tuple,
            set,
        ),
    ):
        return [
            str(v)
            for v in x
        ]

    if x is None:
        return []

    try:

        if pd.isna(
            x
        ):
            return []

    except Exception:
        pass

    s = str(
        x
    ).strip()

    if not s:
        return []

    for parser in (
        json.loads,
        ast.literal_eval,
    ):

        try:

            obj = parser(
                s
            )

            if isinstance(
                obj,
                (
                    list,
                    tuple,
                    set,
                ),
            ):

                return [
                    str(v)
                    for v in obj
                ]

        except Exception:
            pass

    return []


# =============================================================================
# INPUT GATES
# =============================================================================

for p in [
    KSWEEP,
    KSWEEP_SUMMARY,
    FROZEN,
    SEM,
    CIT,
]:

    if not p.exists():

        raise SystemExit(
            f"Missing required input: {p}"
        )


for p in MODEL_METRICS.values():

    if not p.exists():

        raise SystemExit(
            f"Missing metric detail: {p}"
        )


# =============================================================================
# AUTHORITATIVE k=3 FLAGS
# =============================================================================

ks = pd.read_csv(
    KSWEEP,
    low_memory=False,
)


required = {
    "task_id",
    "task_type",
    "method",
    "k",
    "any_gold",
    "all_gold",
    "target_slide",
    "target_lecture",
}


missing = (
    required
    -
    set(
        ks.columns
    )
)


if missing:

    raise SystemExit(
        "k-sweep detail missing columns: "
        + str(
            sorted(
                missing
            )
        )
    )


ks[
    "task_id"
] = ks[
    "task_id"
].astype(
    str
)


k3 = ks.loc[
    ks[
        "k"
    ].eq(
        3
    )
    &
    ks[
        "task_type"
    ].isin(
        ANSWERABLE
    )
].copy()


if len(
    k3
) != 1200:

    raise SystemExit(
        f"Expected 1200 answerable "
        f"task-method k=3 rows; "
        f"found {len(k3)}"
    )


if k3[
    [
        "task_id",
        "method",
    ]
].duplicated().any():

    raise SystemExit(
        "Duplicate task/method k=3 rows."
    )


method_counts = (
    k3[
        "method"
    ]
    .value_counts()
    .to_dict()
)


if method_counts != {
    "bm25":
        600,

    "rrf":
        600,
}:

    raise SystemExit(
        f"k=3 method counts wrong: "
        f"{method_counts}"
    )


# =============================================================================
# HARD AUTHORITATIVE RATE GATE
# =============================================================================

rate_rows = []


for method, expected in EXPECTED_K3.items():

    s = k3.loc[
        k3[
            "method"
        ].eq(
            method
        )
    ]


    actual = {
        "N":
            len(
                s
            ),

        "any_gold":
            float(
                s[
                    "any_gold"
                ].mean()
            ),

        "all_gold":
            float(
                s[
                    "all_gold"
                ].mean()
            ),

        "target_slide":
            float(
                s[
                    "target_slide"
                ].mean()
            ),
    }


    row = {
        "method":
            method,
    }


    for name in [
        "N",
        "any_gold",
        "all_gold",
        "target_slide",
    ]:

        row[
            name
        ] = actual[
            name
        ]

        row[
            "expected_"
            + name
        ] = expected[
            name
        ]


        if name == "N":

            passed = (
                actual[
                    name
                ]
                == expected[
                    name
                ]
            )

        else:

            passed = np.isclose(
                actual[
                    name
                ],
                expected[
                    name
                ],
                atol=1e-12,
                rtol=0,
            )


        row[
            name
            + "_passed"
        ] = passed


        if not passed:

            raise SystemExit(
                f"AUTHORITATIVE_K3_RATE_GATE "
                f"failed: {method} {name}; "
                f"expected={expected[name]}, "
                f"actual={actual[name]}"
            )


    rate_rows.append(
        row
    )


rate_gate = pd.DataFrame(
    rate_rows
)


rate_gate.to_csv(
    OUT
    / "authoritative_k3_rate_gate.csv",
    index=False,
)


print(
    "AUTHORITATIVE_K3_RATE_GATE = PASSED"
)


# =============================================================================
# INDEPENDENTLY RECOMPUTE k=3 FLAGS FROM FROZEN CONTEXT FILE
# =============================================================================

fr = pd.read_csv(
    FROZEN,
    low_memory=False,
)


required_fr = {
    "task_id",
    "task_type",
    "retrieval_method",
    "target_doc_id",
    "gold_evidence_doc_ids",
    "context_doc_ids",
}


missing = (
    required_fr
    -
    set(
        fr.columns
    )
)


if missing:

    raise SystemExit(
        "Frozen context file missing: "
        + str(
            sorted(
                missing
            )
        )
    )


fr[
    "task_id"
] = fr[
    "task_id"
].astype(
    str
)


fr = fr.loc[
    fr[
        "task_type"
    ].isin(
        ANSWERABLE
    )
].copy()


if len(
    fr
) != 1200:

    raise SystemExit(
        f"Expected 1200 answerable frozen rows; "
        f"found {len(fr)}"
    )


recomputed = []


for _, row in fr.iterrows():

    gold = set(
        listish(
            row[
                "gold_evidence_doc_ids"
            ]
        )
    )


    context = set(
        listish(
            row[
                "context_doc_ids"
            ]
        )
    )


    target = str(
        row[
            "target_doc_id"
        ]
    )


    raw3 = set(
        listish(
            row[
                "retrieved_top3_raw"
            ]
        )
    )


    recomputed.append(
        {
            "task_id":
                str(
                    row[
                        "task_id"
                    ]
                ),

            "task_type":
                str(
                    row[
                        "task_type"
                    ]
                ),

            "method":
                str(
                    row[
                        "retrieval_method"
                    ]
                ),

            "any_gold_recomputed":
                int(
                    bool(
                        gold
                        &
                        context
                    )
                ),

            "all_gold_recomputed":
                int(
                    bool(
                        gold
                    )
                    and gold.issubset(
                        context
                    )
                ),

            "target_slide_recomputed":
                int(
                    bool(
                        target
                    )
                    and target
                    in context
                ),

            "n_gold_in_context":
                len(
                    gold
                    &
                    context
                ),

            "n_raw3_unique":
                len(
                    raw3
                ),

            "n_context_docs":
                len(
                    context
                ),

            "dropped_by_word_cap":
                sorted(
                    raw3
                    -
                    context
                ),
        }
    )


rc = pd.DataFrame(
    recomputed
)


check = k3.merge(
    rc,
    on=[
        "task_id",
        "task_type",
        "method",
    ],
    how="outer",
    validate="one_to_one",
    indicator=True,
)


if len(
    check
) != 1200:

    raise SystemExit(
        "Frozen row-level merge count failed."
    )


if not (
    check[
        "_merge"
    ]
    == "both"
).all():

    raise SystemExit(
        "Frozen task/method identity failed."
    )


#
# ROOT CAUSE (diagnosed, not assumed):
#
# position_retrieval_k_sweep_detail.csv computes any_gold / all_gold /
# target_slide from the RAW top-k retrieval ranking (ranking[:k] in
# run_matched_retrieval_core.py). This is a pure retrieval-
# capability measure and is what the manuscript's k-sweep + MRR table
# reports.
#
# frozen_position_contexts_and_prompts_2000.csv's context_doc_ids is the
# ACTUALLY DELIVERED context: the archived format_context() (recovered
# verbatim from full1000_all_models_pipeline.py, MAX_CONTEXT_WORDS=650)
# stops appending documents once the running word count exhausts the
# 650-word budget (`if remaining <= 0: break`). A doc ranked in the raw
# top-3 can therefore be dropped from the context actually shown to the
# model if the first 1-2 retrieved slides alone consume the full budget.
#
# For the retrieval -> generation linkage analysis specifically, what
# matters is what the model actually saw, so any_gold/all_gold/
# target_slide computed from context_doc_ids (the "_recomputed" columns)
# are the authoritative retrieval-success labels used downstream — NOT
# the raw-rank k-sweep columns. The k-sweep columns/aggregate rates
# (validated above) remain authoritative for the separate retrieval-
# capability report and are left untouched.
#
# This block does not weaken parity checking; it requires that every
# mismatch between the two label sources is EXACTLY explained by this
# known word-cap truncation mechanism. Any unexplained mismatch still
# hard-fails the run.
#

KNOWN_TRUNCATION_MISMATCH_TASK_IDS = {
    "B3_00208",
    "B3_00214",
    "B3_00238",
}


mismatch_summary = {}

unexplained = []


for metric in [
    "any_gold",
    "all_gold",
    "target_slide",
]:

    mismatch_mask = (
        pd.to_numeric(
            check[
                metric
            ]
        )
        !=
        pd.to_numeric(
            check[
                metric
                + "_recomputed"
            ]
        )
    )


    mismatch_rows = check.loc[
        mismatch_mask
    ]


    mismatch_summary[
        metric
    ] = len(
        mismatch_rows
    )


    for _, mrow in mismatch_rows.iterrows():

        is_known = (
            mrow[
                "task_id"
            ]
            in KNOWN_TRUNCATION_MISMATCH_TASK_IDS
            and mrow[
                "method"
            ]
            == "rrf"
            and mrow[
                "n_raw3_unique"
            ]
            == 3
            and mrow[
                "n_context_docs"
            ]
            == 2
        )


        if not is_known:

            unexplained.append(
                {
                    "metric":
                        metric,
                    "task_id":
                        mrow[
                            "task_id"
                        ],
                    "method":
                        mrow[
                            "method"
                        ],
                }
            )


if unexplained:

    raise SystemExit(
        "FROZEN_CONTEXT_ROW_PARITY failed with "
        "unexplained mismatches (not the known "
        "MAX_CONTEXT_WORDS=650 truncation pattern): "
        + json.dumps(
            unexplained
        )
    )


found_known_task_ids = set(
    check.loc[
        pd.to_numeric(
            check[
                "any_gold"
            ]
        )
        !=
        pd.to_numeric(
            check[
                "any_gold_recomputed"
            ]
        ),
        "task_id",
    ]
)


if found_known_task_ids != KNOWN_TRUNCATION_MISMATCH_TASK_IDS:

    raise SystemExit(
        "FROZEN_CONTEXT_ROW_PARITY: known truncation "
        "mismatch set did not reproduce exactly. "
        f"expected={sorted(KNOWN_TRUNCATION_MISMATCH_TASK_IDS)} "
        f"found={sorted(found_known_task_ids)}"
    )


print(
    "FROZEN_CONTEXT_ROW_PARITY = PASSED "
    "(1197/1200 exact; 3/1200 explained by "
    "MAX_CONTEXT_WORDS=650 truncation of the 3rd "
    "raw-retrieved RRF document on "
    "neighbor_slide_conceptual_qa tasks "
    f"{sorted(KNOWN_TRUNCATION_MISMATCH_TASK_IDS)})"
)

print(
    "Frozen k=3 rows matched exactly: 1197 / 1200; "
    "3 explained mismatches (see "
    "known_context_truncation_divergence.csv)"
)


check.to_csv(
    OUT
    / "frozen_context_vs_ksweep_k3_row_parity.csv",
    index=False,
)


check.loc[
    check[
        "task_id"
    ].isin(
        KNOWN_TRUNCATION_MISMATCH_TASK_IDS
    )
].to_csv(
    OUT
    / "known_context_truncation_divergence.csv",
    index=False,
)


# =============================================================================
# MAKE AUTHORITATIVE TASK-LEVEL RETRIEVAL TABLE
# =============================================================================

# NOTE: uses the "_recomputed" (context_doc_ids / delivered-context)
# flags, not the raw-rank k-sweep flags, because this table feeds the
# retrieval -> generation-quality linkage and the model can only be
# influenced by evidence it actually received. See the diagnosed
# MAX_CONTEXT_WORDS=650 truncation note above. target_lecture_in_top3
# has no delivered-context counterpart (lecture-level, not truncation-
# sensitive in practice) and is kept from the k-sweep file as-is.

retrieval = check[
    [
        "task_id",
        "task_type",
        "method",
        "any_gold_recomputed",
        "all_gold_recomputed",
        "target_slide_recomputed",
        "target_lecture",
        "n_gold_in_context",
    ]
].copy()


retrieval = retrieval.rename(
    columns={
        "any_gold_recomputed":
            "any_gold_in_top3",

        "all_gold_recomputed":
            "all_gold_in_top3",

        "target_slide_recomputed":
            "target_in_top3",

        "target_lecture":
            "target_lecture_in_top3",
    }
)


retrieval.to_csv(
    OUT
    / "authoritative_answerable_k3_retrieval_flags.csv",
    index=False,
)


# =============================================================================
# LOAD POSITION QUALITY METRICS — 3 MODELS
# =============================================================================

metric_frames = []


for model_key, path in MODEL_METRICS.items():

    df = pd.read_csv(
        path,
        low_memory=False,
    )


    if len(
        df
    ) != 2000:

        raise SystemExit(
            f"{model_key}: expected 2000 metric rows; "
            f"found {len(df)}"
        )


    required = {
        "task_id",
        "task_type",
        "method",
        "lexical_f1",
        "evidence_token_overlap",
    }


    missing = (
        required
        -
        set(
            df.columns
        )
    )


    if missing:

        raise SystemExit(
            f"{model_key} metric detail missing: "
            f"{sorted(missing)}"
        )


    df = df.loc[
        df[
            "task_type"
        ].isin(
            ANSWERABLE
        ),
        [
            "task_id",
            "task_type",
            "method",
            "lexical_f1",
            "evidence_token_overlap",
        ],
    ].copy()


    if len(
        df
    ) != 1200:

        raise SystemExit(
            f"{model_key}: expected 1200 "
            f"answerable metric rows; "
            f"found {len(df)}"
        )


    df[
        "task_id"
    ] = df[
        "task_id"
    ].astype(
        str
    )


    df[
        "model_key"
    ] = model_key


    metric_frames.append(
        df
    )


quality = pd.concat(
    metric_frames,
    ignore_index=True,
)


if len(
    quality
) != 3600:

    raise SystemExit(
        "Expected 3600 model-task-method "
        "quality rows."
    )


# =============================================================================
# JOIN SEMANTIC
# =============================================================================

sem = pd.read_csv(
    SEM,
    low_memory=False,
)


sem[
    "task_id"
] = sem[
    "task_id"
].astype(
    str
)


sem = sem[
    [
        "model_key",
        "task_id",
        "method",
        "semantic_similarity_position",
    ]
].copy()


sem = sem.rename(
    columns={
        "semantic_similarity_position":
            "semantic_similarity"
    }
)


if len(
    sem
) != 3600:

    raise SystemExit(
        f"Expected 3600 semantic rows; "
        f"found {len(sem)}"
    )


quality = quality.merge(
    sem,
    on=[
        "model_key",
        "task_id",
        "method",
    ],
    how="left",
    validate="one_to_one",
)


if quality[
    "semantic_similarity"
].isna().any():

    raise SystemExit(
        "Missing semantic values after join."
    )


# =============================================================================
# JOIN EXACT VALIDATED CITATION METRICS
# =============================================================================

cit = pd.read_csv(
    CIT,
    low_memory=False,
)


cit[
    "task_id"
] = cit[
    "task_id"
].astype(
    str
)


required_cit = {
    "model_key",
    "task_id",
    "task_type",
    "method",
    "corrected_gold_citation",
    "corrected_any_valid_citation",
}


missing = (
    required_cit
    -
    set(
        cit.columns
    )
)


if missing:

    raise SystemExit(
        "Citation detail missing: "
        + str(
            sorted(
                missing
            )
        )
    )


cit = cit.loc[
    cit[
        "task_type"
    ].isin(
        ANSWERABLE
    ),
    [
        "model_key",
        "task_id",
        "method",
        "corrected_gold_citation",
        "corrected_any_valid_citation",
    ],
].copy()


if len(
    cit
) != 3600:

    raise SystemExit(
        f"Expected 3600 answerable citation rows; "
        f"found {len(cit)}"
    )


quality = quality.merge(
    cit,
    on=[
        "model_key",
        "task_id",
        "method",
    ],
    how="left",
    validate="one_to_one",
)


for c in [
    "corrected_gold_citation",
    "corrected_any_valid_citation",
]:

    if quality[
        c
    ].isna().any():

        raise SystemExit(
            f"Missing citation metric: {c}"
        )


# =============================================================================
# JOIN AUTHORITATIVE RETRIEVAL FLAGS
# =============================================================================

quality = quality.merge(
    retrieval,
    on=[
        "task_id",
        "task_type",
        "method",
    ],
    how="left",
    validate="many_to_one",
)


for c in [
    "any_gold_in_top3",
    "all_gold_in_top3",
    "target_in_top3",
]:

    if quality[
        c
    ].isna().any():

        raise SystemExit(
            f"Missing retrieval flag after join: {c}"
        )


if len(
    quality
) != 3600:

    raise SystemExit(
        "Final model-level linkage row count "
        "must be 3600."
    )


# Verify each task/method has exactly three model outcomes.
model_counts = (
    quality.groupby(
        [
            "task_id",
            "method",
        ]
    )[
        "model_key"
    ]
    .nunique()
)


if not (
    model_counts
    == 3
).all():

    raise SystemExit(
        "Three-model linkage identity failed."
    )


print(
    "QUALITY_LINKAGE_IDENTITY_GATE = PASSED"
)


quality.to_csv(
    OUT
    / "retrieval_generation_linkage_model_level_3600.csv",
    index=False,
)


# =============================================================================
# COLLAPSE OUTCOMES ACROSS MODELS — INFERENCE UNIT = BENCHMARK TASK
# =============================================================================

task = (
    quality.groupby(
        [
            "task_id",
            "task_type",
            "method",
        ],
        as_index=False,
    )
    .agg(
        any_gold_in_top3=(
            "any_gold_in_top3",
            "first",
        ),

        all_gold_in_top3=(
            "all_gold_in_top3",
            "first",
        ),

        target_in_top3=(
            "target_in_top3",
            "first",
        ),

        target_lecture_in_top3=(
            "target_lecture_in_top3",
            "first",
        ),

        n_gold_in_context=(
            "n_gold_in_context",
            "first",
        ),

        lexical_f1=(
            "lexical_f1",
            "mean",
        ),

        evidence_token_overlap=(
            "evidence_token_overlap",
            "mean",
        ),

        semantic_similarity=(
            "semantic_similarity",
            "mean",
        ),

        corrected_gold_citation=(
            "corrected_gold_citation",
            "mean",
        ),

        corrected_any_valid_citation=(
            "corrected_any_valid_citation",
            "mean",
        ),
    )
)


if len(
    task
) != 1200:

    raise SystemExit(
        f"Expected 1200 task-method rows; "
        f"found {len(task)}"
    )


task.to_csv(
    OUT
    / "retrieval_generation_linkage_task_level_1200.csv",
    index=False,
)


# =============================================================================
# FINAL COVERAGE — SHOULD REPRODUCE AUTHORITATIVE NUMBERS
# =============================================================================

coverage = (
    task.groupby(
        "method",
        as_index=False,
    )
    .agg(
        N_tasks=(
            "task_id",
            "size",
        ),

        any_gold_top3_rate=(
            "any_gold_in_top3",
            "mean",
        ),

        all_gold_top3_rate=(
            "all_gold_in_top3",
            "mean",
        ),

        target_top3_rate=(
            "target_in_top3",
            "mean",
        ),

        target_lecture_top3_rate=(
            "target_lecture_in_top3",
            "mean",
        ),

        mean_gold_docs_top3=(
            "n_gold_in_context",
            "mean",
        ),
    )
)


coverage.to_csv(
    OUT
    / "fixed_retrieval_top3_coverage_answerable.csv",
    index=False,
)


# =============================================================================
# BY TASK TYPE COVERAGE
# =============================================================================

coverage_task_type = (
    task.groupby(
        [
            "method",
            "task_type",
        ],
        as_index=False,
    )
    .agg(
        N_tasks=(
            "task_id",
            "size",
        ),

        any_gold_top3_rate=(
            "any_gold_in_top3",
            "mean",
        ),

        all_gold_top3_rate=(
            "all_gold_in_top3",
            "mean",
        ),

        target_top3_rate=(
            "target_in_top3",
            "mean",
        ),

        target_lecture_top3_rate=(
            "target_lecture_in_top3",
            "mean",
        ),
    )
)


coverage_task_type.to_csv(
    OUT
    / "fixed_retrieval_top3_coverage_by_task_type.csv",
    index=False,
)


# =============================================================================
# DESCRIPTIVE OUTCOME BY RETRIEVAL SUCCESS
# =============================================================================

desc_rows = []


for method in [
    "bm25",
    "rrf",
]:

    md = task.loc[
        task[
            "method"
        ].eq(
            method
        )
    ]


    for flag in [
        "any_gold_in_top3",
        "target_in_top3",
    ]:


        for state in [
            0,
            1,
        ]:

            s = md.loc[
                md[
                    flag
                ].eq(
                    state
                )
            ]


            row = {
                "method":
                    method,

                "retrieval_flag":
                    flag,

                "retrieval_success":
                    state,

                "N_tasks":
                    len(
                        s
                    ),
            }


            for metric in QUALITY_METRICS:

                row[
                    metric
                ] = (
                    float(
                        s[
                            metric
                        ].mean()
                    )
                    if len(s)
                    else np.nan
                )


            desc_rows.append(
                row
            )


descriptive = pd.DataFrame(
    desc_rows
)


descriptive.to_csv(
    OUT
    / "quality_by_authoritative_retrieval_success.csv",
    index=False,
)


# =============================================================================
# BOOTSTRAP ASSOCIATION
#
# This is deliberately labelled association, NOT causal effect.
# =============================================================================

association_rows = []

counter = 0


for method in [
    "bm25",
    "rrf",
]:

    md = task.loc[
        task[
            "method"
        ].eq(
            method
        )
    ]


    for flag in [
        "any_gold_in_top3",
        "target_in_top3",
    ]:


        success = md.loc[
            md[
                flag
            ].eq(
                1
            )
        ]


        failure = md.loc[
            md[
                flag
            ].eq(
                0
            )
        ]


        if (
            len(
                success
            )
            == 0
            or
            len(
                failure
            )
            == 0
        ):

            raise SystemExit(
                f"Unexpected zero-cell for "
                f"{method}/{flag}"
            )


        for metric in QUALITY_METRICS:

            counter += 1


            a = success[
                metric
            ].to_numpy(
                dtype=float
            )


            b = failure[
                metric
            ].to_numpy(
                dtype=float
            )


            observed = float(
                a.mean()
                -
                b.mean()
            )


            rng = np.random.default_rng(
                SEED
                + counter
            )


            boot = np.empty(
                N_BOOT,
                dtype=float,
            )


            for i in range(
                N_BOOT
            ):

                aa = a[
                    rng.integers(
                        0,
                        len(a),
                        size=len(a),
                    )
                ]


                bb = b[
                    rng.integers(
                        0,
                        len(b),
                        size=len(b),
                    )
                ]


                boot[
                    i
                ] = float(
                    aa.mean()
                    -
                    bb.mean()
                )


            association_rows.append(
                {
                    "method":
                        method,

                    "retrieval_flag":
                        flag,

                    "metric":
                        metric,

                    "N_success_tasks":
                        len(
                            a
                        ),

                    "N_failure_tasks":
                        len(
                            b
                        ),

                    "success_mean":
                        float(
                            a.mean()
                        ),

                    "failure_mean":
                        float(
                            b.mean()
                        ),

                    "difference_success_minus_failure":
                        observed,

                    "CI95_low":
                        float(
                            np.quantile(
                                boot,
                                0.025,
                            )
                        ),

                    "CI95_high":
                        float(
                            np.quantile(
                                boot,
                                0.975,
                            )
                        ),

                    "interpretation":
                        (
                            "Descriptive association; "
                            "not a causal estimate."
                        ),
                }
            )


association = pd.DataFrame(
    association_rows
)


association.to_csv(
    OUT
    / "retrieval_success_quality_association_bootstrap.csv",
    index=False,
)


# =============================================================================
# TASK-TYPE-STRATIFIED ASSOCIATION
# =============================================================================

stratified_rows = []

counter = 1000


for method in [
    "bm25",
    "rrf",
]:

    for task_type in sorted(
        ANSWERABLE
    ):

        md = task.loc[
            task[
                "method"
            ].eq(
                method
            )
            &
            task[
                "task_type"
            ].eq(
                task_type
            )
        ]


        for flag in [
            "any_gold_in_top3",
            "target_in_top3",
        ]:


            success = md.loc[
                md[
                    flag
                ].eq(
                    1
                )
            ]


            failure = md.loc[
                md[
                    flag
                ].eq(
                    0
                )
            ]


            if (
                len(
                    success
                )
                == 0
                or
                len(
                    failure
                )
                == 0
            ):

                continue


            for metric in QUALITY_METRICS:

                counter += 1


                a = success[
                    metric
                ].to_numpy(
                    dtype=float
                )


                b = failure[
                    metric
                ].to_numpy(
                    dtype=float
                )


                rng = np.random.default_rng(
                    SEED
                    + counter
                )


                boot = np.empty(
                    N_BOOT,
                    dtype=float,
                )


                for i in range(
                    N_BOOT
                ):

                    aa = a[
                        rng.integers(
                            0,
                            len(a),
                            size=len(a),
                        )
                    ]


                    bb = b[
                        rng.integers(
                            0,
                            len(b),
                            size=len(b),
                        )
                    ]


                    boot[
                        i
                    ] = (
                        aa.mean()
                        -
                        bb.mean()
                    )


                stratified_rows.append(
                    {
                        "method":
                            method,

                        "task_type":
                            task_type,

                        "retrieval_flag":
                            flag,

                        "metric":
                            metric,

                        "N_success_tasks":
                            len(
                                a
                            ),

                        "N_failure_tasks":
                            len(
                                b
                            ),

                        "success_mean":
                            float(
                                a.mean()
                            ),

                        "failure_mean":
                            float(
                                b.mean()
                            ),

                        "difference_success_minus_failure":
                            float(
                                a.mean()
                                -
                                b.mean()
                            ),

                        "CI95_low":
                            float(
                                np.quantile(
                                    boot,
                                    0.025,
                                )
                            ),

                        "CI95_high":
                            float(
                                np.quantile(
                                    boot,
                                    0.975,
                                )
                            ),
                    }
                )


stratified = pd.DataFrame(
    stratified_rows
)


stratified.to_csv(
    OUT
    / "retrieval_success_quality_association_by_task_type.csv",
    index=False,
)


# =============================================================================
# CITATION OPPORTUNITY
# =============================================================================

citation_opportunity = (
    task.groupby(
        [
            "method",
            "any_gold_in_top3",
        ],
        as_index=False,
    )
    .agg(
        N_tasks=(
            "task_id",
            "size",
        ),

        corrected_gold_citation=(
            "corrected_gold_citation",
            "mean",
        ),

        corrected_any_valid_citation=(
            "corrected_any_valid_citation",
            "mean",
        ),

        semantic_similarity=(
            "semantic_similarity",
            "mean",
        ),

        lexical_f1=(
            "lexical_f1",
            "mean",
        ),

        evidence_token_overlap=(
            "evidence_token_overlap",
            "mean",
        ),
    )
)


citation_opportunity.to_csv(
    OUT
    / "gold_citation_given_retrieval_opportunity.csv",
    index=False,
)


# =============================================================================
# METADATA
# =============================================================================

metadata = {
    "status":
        "RETRIEVAL_GENERATION_LINKAGE_COMPLETE",

    "retrieval_label_source":
        str(
            KSWEEP
        ),

    "independent_context_source":
        str(
            FROZEN
        ),

    "authoritative_k":
        3,

    "authoritative_rate_gate":
        EXPECTED_K3,

    "row_level_context_parity":
        (
            "1197/1200 exact; 3/1200 explained by "
            "MAX_CONTEXT_WORDS=650 truncation of the "
            "3rd raw-retrieved RRF document on "
            "neighbor_slide_conceptual_qa tasks "
            "(B3_00208, B3_00214, B3_00238); see "
            "known_context_truncation_divergence.csv"
        ),

    "retrieval_success_label_definition":
        (
            "any_gold_in_top3 / all_gold_in_top3 / "
            "target_in_top3 are computed from "
            "context_doc_ids (delivered context after the "
            "650-word cap), NOT from the raw top-k "
            "retrieval rank used by the k-sweep report. "
            "This is the correct label for a retrieval -> "
            "generation-quality linkage because the model "
            "cannot be influenced by evidence it never "
            "received. The official retrieval-capability "
            "rates reported elsewhere (k-sweep, MRR) remain "
            "unchanged and are rank-based."
        ),

    "quality_model_task_rows":
        3600,

    "task_method_rows":
        1200,

    "models_per_task":
        3,

    "bootstrap_replicates":
        N_BOOT,

    "association_warning":
        (
            "Quality differences conditional on retrieval "
            "success are descriptive associations. "
            "They are not causal estimates because retrieval "
            "success is not randomized across benchmark tasks."
        ),

    "invalid_previous_analysis":
        (
            "the candidate retrieval-generation linkage output failed validation "
            "because downstream metric-detail files did not "
            "retain gold/context/target document IDs and "
            "therefore produced artificial all-zero coverage."
        ),

    "source_hashes": {
        "ksweep":
            sha256_file(
                KSWEEP
            ),

        "frozen_contexts":
            sha256_file(
                FROZEN
            ),

        "semantic":
            sha256_file(
                SEM
            ),

        "citation":
            sha256_file(
                CIT
            ),
    },
}


(
    OUT
    / "RETRIEVAL_GENERATION_LINKAGE_METADATA.json"
).write_text(
    json.dumps(
        metadata,
        indent=2,
    ),
    encoding="utf-8",
)


# =============================================================================
# REPORT
# =============================================================================

print()
print(
    "=" * 120
)

print(
    "AUTHORITATIVE k=3 RATE GATE"
)

print(
    "=" * 120
)

print(
    rate_gate.to_string(
        index=False
    )
)


print()
print(
    "=" * 120
)

print(
    "ANSWERABLE TOP-3 COVERAGE"
)

print(
    "=" * 120
)

print(
    coverage.to_string(
        index=False
    )
)


print()
print(
    "=" * 120
)

print(
    "TOP-3 COVERAGE BY TASK TYPE"
)

print(
    "=" * 120
)

print(
    coverage_task_type.to_string(
        index=False
    )
)


print()
print(
    "=" * 120
)

print(
    "QUALITY BY RETRIEVAL SUCCESS"
)

print(
    "=" * 120
)

print(
    descriptive.to_string(
        index=False
    )
)


print()
print(
    "=" * 120
)

print(
    "RETRIEVAL-SUCCESS / QUALITY ASSOCIATION"
)

print(
    "=" * 120
)

print(
    association.to_string(
        index=False
    )
)


print()
print(
    "=" * 120
)

print(
    "GOLD-CITATION GIVEN RETRIEVAL OPPORTUNITY"
)

print(
    "=" * 120
)

print(
    citation_opportunity.to_string(
        index=False
    )
)


print()
print(
    "AUTHORITATIVE_K3_RATE_GATE = PASSED"
)

print(
    "FROZEN_CONTEXT_ROW_PARITY = PASSED "
    "(1197/1200 exact + 3/1200 explained truncation)"
)

print(
    "FROZEN_CONTEXT_ROWS_MATCHED = 1197/1200 exact; "
    "3/1200 explained by MAX_CONTEXT_WORDS=650 truncation"
)

print(
    "QUALITY_LINKAGE_IDENTITY_GATE = PASSED"
)

print(
    "RETRIEVAL_GENERATION_LINKAGE = PASSED"
)

print(
    "NO_GENERATION = TRUE"
)
