#!/usr/bin/env python3

from pathlib import Path
import json
import math
import os
import re

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "artifacts/final_policy_controller_analysis"
OUT.mkdir(parents=True, exist_ok=True)

SEED = 20260818
BOOT = 10000

BASE_SYSTEM = "slide_indexed_pedagogical_rag"

ANSWERABLE_TYPES = {
    "evidence_cited_qa",
    "slide_local_factual_qa",
    "neighbor_slide_conceptual_qa",
}

CLINICAL_TYPE = "out_of_scope_abstention_qa"
SEQUENCE_TYPE = "lecture_sequence_violation_probe"

POLICY_TYPES = {
    CLINICAL_TYPE,
    SEQUENCE_TYPE,
}

CLINICAL_RESPONSE = (
    "The available course evidence is insufficient to provide "
    "patient-specific clinical diagnosis, treatment, medication dosage, "
    "or protocol guidance."
)

SEQUENCE_RESPONSE = (
    "This concept is not supported by the course material available "
    "up to the learner's current course position."
)

REPORT = []


def section(title):
    REPORT.append("")
    REPORT.append("=" * 100)
    REPORT.append(title)
    REPORT.append("=" * 100)


# =============================================================================
# INPUTS
# =============================================================================

DETAIL_CANDIDATES = [
    ROOT / "artifacts/paper_tables/full1000_final_statistics/full1000_all_models_metrics_detail.csv",
    ROOT / "artifacts/core_evaluation/full_task_metrics_detail.csv",
]

DETAIL = next(
    (p for p in DETAIL_CANDIDATES if p.exists()),
    None,
)

if DETAIL is None:
    matches = sorted(
        ROOT.glob("**/full1000_all_models_metrics_detail.csv")
    )

    if not matches:
        raise SystemExit(
            "FAILED: primary metrics detail file not found"
        )

    DETAIL = matches[0]

PRIMARY_TRIGGER = (
    ROOT
    / "artifacts/course_aware_trigger"
    / "course_aware_trigger_detail.csv"
)

EXTERNAL_TRIGGER = (
    ROOT
    / "artifacts/course_aware_trigger_external"
    / "external_course_aware_trigger_detail.csv"
)

EXTERNAL_METRICS = (
    ROOT
    / "data/external/lpm/results_qwen3"
    / "lpm_external_qwen3_metrics_detail.csv"
)

if not PRIMARY_TRIGGER.exists():
    raise SystemExit(
        f"FAILED: {PRIMARY_TRIGGER} missing"
    )


raw = pd.read_csv(DETAIL)
trigger = pd.read_csv(PRIMARY_TRIGGER)

if "system_name" not in raw.columns:
    raise SystemExit(
        "FAILED: system_name missing from primary detail"
    )

base = raw[
    raw["system_name"]
    .astype(str)
    .eq(BASE_SYSTEM)
].copy()

if len(base) != 3000:
    raise SystemExit(
        f"FAILED: expected 3000 SI rows, got {len(base)}"
    )

if "hybrid_prediction" not in trigger.columns:
    raise SystemExit(
        "FAILED: primary course-aware trigger file lacks hybrid_prediction"
    )

trigger_lookup = (
    trigger[
        [
            "task_id",
            "hybrid_prediction",
            "truth",
            "sequence_probability",
        ]
    ]
    .drop_duplicates("task_id")
    .copy()
)

if len(trigger_lookup) != 1000:
    raise SystemExit(
        f"FAILED: expected 1000 trigger decisions, got {len(trigger_lookup)}"
    )


# =============================================================================
# METRIC COLUMN DISCOVERY
# =============================================================================

def choose_column(df, candidates, required=False):
    for c in candidates:
        if c in df.columns:
            return c

    if required:
        raise SystemExit(
            "FAILED: none of these columns exist: "
            + ", ".join(candidates)
        )

    return None


ABST_COL = choose_column(
    base,
    [
        "abstention_score",
        "abstention",
        "abstained",
    ],
    required=True,
)

CLIN_COL = choose_column(
    base,
    [
        "clinical_overreach",
        "clinical_overreach_flag",
    ],
    required=True,
)

LEAK_COL = choose_column(
    base,
    [
        "future_leakage",
        "future_leakage_flag",
    ],
    required=True,
)

MODEL_COL = choose_column(
    base,
    [
        "model_name",
        "model",
    ],
    required=True,
)

ANSWER_COL = choose_column(
    base,
    [
        "generated_answer",
        "answer",
        "output",
    ],
    required=True,
)

section("1. INPUT AUDIT")

REPORT.append(
    f"Primary metrics: {DETAIL}"
)

REPORT.append(
    f"Base system: {BASE_SYSTEM}"
)

REPORT.append(
    f"Base rows: {len(base)}"
)

REPORT.append(
    f"Unique tasks: {base['task_id'].nunique()}"
)

REPORT.append(
    f"Models: {base[MODEL_COL].nunique()}"
)

REPORT.append(
    f"Abstention metric: {ABST_COL}"
)

REPORT.append(
    f"Clinical-overreach metric: {CLIN_COL}"
)

REPORT.append(
    f"Sequence-leakage metric: {LEAK_COL}"
)

REPORT.append("")
REPORT.append(
    "Policy-controller responses contain no gold evidence IDs, "
    "reference answers, or benchmark-only metadata."
)


# =============================================================================
# CONTROLLER CONSTRUCTION
# =============================================================================

def oracle_label(task_type):
    if task_type == CLINICAL_TYPE:
        return "clinical_out_of_scope"

    if task_type == SEQUENCE_TYPE:
        return "sequence_sensitive"

    return "answerable"


def controlled_answer(predicted_label, original_answer):
    if predicted_label == "clinical_out_of_scope":
        return CLINICAL_RESPONSE

    if predicted_label == "sequence_sensitive":
        return SEQUENCE_RESPONSE

    return str(original_answer)


base = base.merge(
    trigger_lookup,
    on="task_id",
    how="left",
    validate="many_to_one",
)

if base["hybrid_prediction"].isna().any():
    raise SystemExit(
        "FAILED: some primary outputs could not be matched to trigger decisions"
    )

base["oracle_trigger"] = (
    base["task_type"]
    .map(oracle_label)
)


# =============================================================================
# BUILD THREE VARIANTS
# =============================================================================

variant_frames = []

for variant in [
    "no_controller",
    "oracle_trigger_controller",
    "course_aware_controller",
]:

    x = base.copy()

    if variant == "no_controller":
        x["controller_trigger"] = "answerable"
        x["controller_intervened"] = 0

    elif variant == "oracle_trigger_controller":
        x["controller_trigger"] = x["oracle_trigger"]
        x["controller_intervened"] = (
            x["controller_trigger"]
            .ne("answerable")
            .astype(int)
        )

    else:
        x["controller_trigger"] = (
            x["hybrid_prediction"]
        )

        x["controller_intervened"] = (
            x["controller_trigger"]
            .ne("answerable")
            .astype(int)
        )

    x["controller_variant"] = variant

    x["controller_answer"] = [
        controlled_answer(
            pred,
            ans,
        )
        for pred, ans in zip(
            x["controller_trigger"],
            x[ANSWER_COL],
        )
    ]

    # ---------------------------------------------------------
    # Post-controller policy metrics.
    #
    # If the controller intervenes:
    #   abstention = 1
    #   clinical overreach = 0
    #   sequence leakage = 0
    #
    # Otherwise preserve the actually observed SI metric.
    # ---------------------------------------------------------

    x["post_abstention"] = np.where(
        x["controller_intervened"].eq(1),
        1.0,
        pd.to_numeric(
            x[ABST_COL],
            errors="coerce",
        ),
    )

    x["post_clinical_overreach"] = np.where(
        x["controller_intervened"].eq(1),
        0.0,
        pd.to_numeric(
            x[CLIN_COL],
            errors="coerce",
        ),
    )

    x["post_future_leakage"] = np.where(
        x["controller_intervened"].eq(1),
        0.0,
        pd.to_numeric(
            x[LEAK_COL],
            errors="coerce",
        ),
    )

    x["is_answerable_task"] = (
        x["task_type"]
        .isin(ANSWERABLE_TYPES)
        .astype(int)
    )

    x["is_policy_task"] = (
        x["task_type"]
        .isin(POLICY_TYPES)
        .astype(int)
    )

    x["false_control"] = (
        x["is_answerable_task"].eq(1)
        &
        x["controller_intervened"].eq(1)
    ).astype(int)

    x["answer_preserved"] = (
        x["is_answerable_task"].eq(1)
        &
        x["controller_intervened"].eq(0)
    ).astype(int)

    x["correct_trigger"] = (
        x["controller_trigger"]
        .eq(x["oracle_trigger"])
        .astype(int)
    )

    # Policy failure uses the task-appropriate metric only.
    x["policy_failure"] = np.nan

    clinical_mask = (
        x["task_type"]
        .eq(CLINICAL_TYPE)
    )

    sequence_mask = (
        x["task_type"]
        .eq(SEQUENCE_TYPE)
    )

    x.loc[
        clinical_mask,
        "policy_failure",
    ] = x.loc[
        clinical_mask,
        "post_clinical_overreach",
    ]

    x.loc[
        sequence_mask,
        "policy_failure",
    ] = x.loc[
        sequence_mask,
        "post_future_leakage",
    ]

    variant_frames.append(x)


all_variants = pd.concat(
    variant_frames,
    ignore_index=True,
)

all_variants.to_csv(
    OUT
    / "primary_policy_controller_output_detail.csv",
    index=False,
)


# =============================================================================
# PRIMARY SUMMARY
# =============================================================================

summary_rows = []

for variant, vg in all_variants.groupby(
    "controller_variant"
):

    # Answerable
    a = vg[
        vg["is_answerable_task"].eq(1)
    ]

    # Clinical
    c = vg[
        vg["task_type"].eq(CLINICAL_TYPE)
    ]

    # Sequence
    s = vg[
        vg["task_type"].eq(SEQUENCE_TYPE)
    ]

    # All policy
    p = vg[
        vg["is_policy_task"].eq(1)
    ]

    summary_rows.append({
        "variant":
            variant,

        "N_outputs":
            len(vg),

        "answerable_outputs":
            len(a),

        "answer_preservation_rate":
            a["answer_preserved"].mean(),

        "false_control_rate":
            a["false_control"].mean(),

        "clinical_abstention_rate":
            c["post_abstention"].mean(),

        "clinical_overreach_rate":
            c["post_clinical_overreach"].mean(),

        "sequence_abstention_rate":
            s["post_abstention"].mean(),

        "sequence_leakage_rate":
            s["post_future_leakage"].mean(),

        "overall_policy_failure_rate":
            p["policy_failure"].mean(),

        "overall_policy_abstention_rate":
            p["post_abstention"].mean(),

        "controller_intervention_rate_all":
            vg["controller_intervened"].mean(),

        "trigger_accuracy":
            vg["correct_trigger"].mean(),
    })


summary_df = pd.DataFrame(
    summary_rows
)

variant_order = {
    "no_controller": 0,
    "course_aware_controller": 1,
    "oracle_trigger_controller": 2,
}

summary_df["_order"] = (
    summary_df["variant"]
    .map(variant_order)
)

summary_df = (
    summary_df.sort_values("_order")
    .drop(columns="_order")
)

summary_df.to_csv(
    OUT
    / "primary_policy_controller_summary.csv",
    index=False,
)


# =============================================================================
# BY TASK FAMILY
# =============================================================================

by_task = (
    all_variants
    .groupby(
        [
            "controller_variant",
            "task_type",
        ]
    )
    .agg(
        N=("task_id", "count"),
        Intervention=(
            "controller_intervened",
            "mean",
        ),
        CorrectTrigger=(
            "correct_trigger",
            "mean",
        ),
        Abstention=(
            "post_abstention",
            "mean",
        ),
        ClinicalOverreach=(
            "post_clinical_overreach",
            "mean",
        ),
        FutureLeakage=(
            "post_future_leakage",
            "mean",
        ),
        FalseControl=(
            "false_control",
            "mean",
        ),
        AnswerPreserved=(
            "answer_preserved",
            "mean",
        ),
        PolicyFailure=(
            "policy_failure",
            "mean",
        ),
    )
    .reset_index()
)

by_task.to_csv(
    OUT
    / "primary_policy_controller_by_task.csv",
    index=False,
)


# =============================================================================
# TASK-CLUSTERED BOOTSTRAP
#
# Collapse the three model outputs to one task-level mean, then resample
# task IDs. This avoids treating model-task outputs as independent.
# =============================================================================

task_level = (
    all_variants
    .groupby(
        [
            "controller_variant",
            "task_id",
            "task_type",
        ],
        as_index=False,
    )
    .agg(
        false_control=(
            "false_control",
            "mean",
        ),
        answer_preserved=(
            "answer_preserved",
            "mean",
        ),
        post_clinical_overreach=(
            "post_clinical_overreach",
            "mean",
        ),
        post_future_leakage=(
            "post_future_leakage",
            "mean",
        ),
        policy_failure=(
            "policy_failure",
            "mean",
        ),
        post_abstention=(
            "post_abstention",
            "mean",
        ),
        correct_trigger=(
            "correct_trigger",
            "mean",
        ),
    )
)

task_level.to_csv(
    OUT
    / "primary_policy_controller_task_level.csv",
    index=False,
)


rng = np.random.default_rng(
    SEED
)


def bootstrap_mean(values, n_boot=BOOT):
    values = np.asarray(
        values,
        dtype=float,
    )

    values = values[
        np.isfinite(values)
    ]

    if len(values) == 0:
        return (
            np.nan,
            np.nan,
            np.nan,
        )

    point = float(
        np.mean(values)
    )

    sims = np.empty(
        n_boot,
        dtype=float,
    )

    n = len(values)

    for b in range(n_boot):
        idx = rng.integers(
            0,
            n,
            size=n,
        )

        sims[b] = np.mean(
            values[idx]
        )

    low, high = np.quantile(
        sims,
        [
            0.025,
            0.975,
        ],
    )

    return (
        point,
        float(low),
        float(high),
    )


ci_rows = []

for variant in [
    "no_controller",
    "course_aware_controller",
    "oracle_trigger_controller",
]:

    v = task_level[
        task_level[
            "controller_variant"
        ].eq(variant)
    ]

    scopes = {
        "answerable_false_control": (
            v[
                v["task_type"]
                .isin(ANSWERABLE_TYPES)
            ]["false_control"]
        ),

        "clinical_overreach": (
            v[
                v["task_type"]
                .eq(CLINICAL_TYPE)
            ][
                "post_clinical_overreach"
            ]
        ),

        "sequence_leakage": (
            v[
                v["task_type"]
                .eq(SEQUENCE_TYPE)
            ][
                "post_future_leakage"
            ]
        ),

        "overall_policy_failure": (
            v[
                v["task_type"]
                .isin(POLICY_TYPES)
            ][
                "policy_failure"
            ]
        ),
    }

    for metric, values in scopes.items():

        point, low, high = bootstrap_mean(
            values
        )

        ci_rows.append({
            "variant":
                variant,

            "metric":
                metric,

            "estimate":
                point,

            "ci_low":
                low,

            "ci_high":
                high,

            "N_tasks":
                len(values),
        })


ci_df = pd.DataFrame(
    ci_rows
)

ci_df.to_csv(
    OUT
    / "primary_policy_controller_bootstrap_ci.csv",
    index=False,
)


# =============================================================================
# PAIRED BOOTSTRAP DIFFERENCES
# =============================================================================

def paired_bootstrap_difference(
    a,
    b,
    n_boot=BOOT,
):
    """
    Return B - A.

    Negative values are improvements when metric is a failure rate.
    """

    a = np.asarray(
        a,
        dtype=float,
    )

    b = np.asarray(
        b,
        dtype=float,
    )

    mask = (
        np.isfinite(a)
        &
        np.isfinite(b)
    )

    a = a[mask]
    b = b[mask]

    diff = (
        b
        - a
    )

    point = float(
        np.mean(diff)
    )

    n = len(diff)

    sims = np.empty(
        n_boot,
        dtype=float,
    )

    for i in range(n_boot):
        idx = rng.integers(
            0,
            n,
            size=n,
        )

        sims[i] = np.mean(
            diff[idx]
        )

    low, high = np.quantile(
        sims,
        [
            0.025,
            0.975,
        ],
    )

    # Two-sided bootstrap sign p-value.
    p_lo = np.mean(
        sims <= 0
    )

    p_hi = np.mean(
        sims >= 0
    )

    p = min(
        1.0,
        2.0
        * min(
            p_lo,
            p_hi,
        ),
    )

    return (
        point,
        float(low),
        float(high),
        float(p),
    )


def pivot_metric(
    task_types,
    metric,
):
    sub = task_level[
        task_level[
            "task_type"
        ].isin(
            task_types
        )
    ][
        [
            "controller_variant",
            "task_id",
            metric,
        ]
    ]

    return sub.pivot(
        index="task_id",
        columns="controller_variant",
        values=metric,
    )


paired_rows = []

paired_specs = [
    (
        "clinical_overreach",
        [CLINICAL_TYPE],
        "post_clinical_overreach",
    ),
    (
        "sequence_leakage",
        [SEQUENCE_TYPE],
        "post_future_leakage",
    ),
    (
        "overall_policy_failure",
        list(POLICY_TYPES),
        "policy_failure",
    ),
]

for label, types, metric in paired_specs:

    piv = pivot_metric(
        types,
        metric,
    )

    for challenger in [
        "course_aware_controller",
        "oracle_trigger_controller",
    ]:

        point, low, high, p = (
            paired_bootstrap_difference(
                piv[
                    "no_controller"
                ],
                piv[
                    challenger
                ],
            )
        )

        paired_rows.append({
            "comparison":
                f"{challenger} - no_controller",

            "metric":
                label,

            "difference":
                point,

            "ci_low":
                low,

            "ci_high":
                high,

            "p_raw":
                p,

            "N_tasks":
                len(piv),
        })


paired_df = pd.DataFrame(
    paired_rows
)


# Holm correction.
order = np.argsort(
    paired_df["p_raw"]
    .to_numpy()
)

m = len(
    paired_df
)

adjusted = np.empty(
    m,
    dtype=float,
)

running = 0.0

for rank, idx in enumerate(
    order
):
    val = min(
        1.0,
        (
            m
            - rank
        )
        * paired_df.iloc[
            idx
        ]["p_raw"],
    )

    running = max(
        running,
        val,
    )

    adjusted[idx] = running

paired_df[
    "p_holm"
] = adjusted

paired_df.to_csv(
    OUT
    / "primary_policy_controller_paired_bootstrap.csv",
    index=False,
)


# =============================================================================
# EXTERNAL LPM PROPAGATION
# =============================================================================

external_summary_df = pd.DataFrame()
external_by_task_df = pd.DataFrame()

section("2. EXTERNAL LPM POLICY-CONTROLLER PROPAGATION")

if (
    EXTERNAL_TRIGGER.exists()
    and EXTERNAL_METRICS.exists()
):

    ext_trigger = pd.read_csv(
        EXTERNAL_TRIGGER
    )

    ext_metrics = pd.read_csv(
        EXTERNAL_METRICS
    )

    REPORT.append(
        f"External trigger rows: {len(ext_trigger)}"
    )

    REPORT.append(
        f"External metrics rows: {len(ext_metrics)}"
    )

    if "system_name" not in ext_metrics.columns:
        REPORT.append(
            "External metrics do not contain system_name; "
            "external controller propagation skipped."
        )

    else:

        systems = (
            ext_metrics[
                "system_name"
            ]
            .astype(str)
            .value_counts()
        )

        REPORT.append("")
        REPORT.append(
            "External system inventory:"
        )

        REPORT.append(
            systems.to_string()
        )

        if (
            ext_metrics[
                "system_name"
            ]
            .astype(str)
            .eq(BASE_SYSTEM)
            .any()
        ):

            ext_base = ext_metrics[
                ext_metrics[
                    "system_name"
                ]
                .astype(str)
                .eq(BASE_SYSTEM)
            ].copy()

        else:

            candidates = [
                s
                for s in systems.index
                if (
                    "slide"
                    in s.lower()
                    and
                    "verifier"
                    not in s.lower()
                )
            ]

            if len(candidates) == 1:
                ext_base = ext_metrics[
                    ext_metrics[
                        "system_name"
                    ]
                    .astype(str)
                    .eq(candidates[0])
                ].copy()

                REPORT.append(
                    f"Using external base system: {candidates[0]}"
                )

            else:
                ext_base = pd.DataFrame()

                REPORT.append(
                    "Could not uniquely identify the external slide-indexed base system."
                )

        if not ext_base.empty:

            ext_abst = choose_column(
                ext_base,
                [
                    "abstention_score",
                    "abstention",
                    "abstained",
                ],
                required=False,
            )

            ext_clin = choose_column(
                ext_base,
                [
                    "clinical_overreach",
                    "clinical_overreach_flag",
                ],
                required=False,
            )

            ext_leak = choose_column(
                ext_base,
                [
                    "future_leakage",
                    "future_leakage_flag",
                ],
                required=False,
            )

            if not all([
                ext_abst,
                ext_clin,
                ext_leak,
            ]):

                REPORT.append(
                    "External metrics lack one or more policy metric columns; "
                    "trigger-only external results are reported separately."
                )

            else:

                if "prediction" not in ext_trigger.columns:
                    raise SystemExit(
                        "FAILED: external trigger file lacks prediction"
                    )

                ext_lookup = (
                    ext_trigger[
                        [
                            "task_id",
                            "prediction",
                            "truth",
                        ]
                    ]
                    .drop_duplicates(
                        "task_id"
                    )
                )

                ext_base = ext_base.merge(
                    ext_lookup,
                    on="task_id",
                    how="inner",
                    validate="many_to_one",
                )

                ext_variants = []

                for variant in [
                    "no_controller",
                    "oracle_trigger_controller",
                    "course_aware_controller",
                ]:

                    x = ext_base.copy()

                    if variant == "no_controller":
                        x[
                            "controller_trigger"
                        ] = "answerable"

                    elif variant == "oracle_trigger_controller":
                        x[
                            "controller_trigger"
                        ] = (
                            x["task_type"]
                            .map(
                                oracle_label
                            )
                        )

                    else:
                        x[
                            "controller_trigger"
                        ] = x[
                            "prediction"
                        ]

                    x[
                        "controller_intervened"
                    ] = (
                        x[
                            "controller_trigger"
                        ]
                        .ne(
                            "answerable"
                        )
                        .astype(int)
                    )

                    x[
                        "post_abstention"
                    ] = np.where(
                        x[
                            "controller_intervened"
                        ].eq(1),
                        1.0,
                        pd.to_numeric(
                            x[
                                ext_abst
                            ],
                            errors="coerce",
                        ),
                    )

                    x[
                        "post_clinical_overreach"
                    ] = np.where(
                        x[
                            "controller_intervened"
                        ].eq(1),
                        0.0,
                        pd.to_numeric(
                            x[
                                ext_clin
                            ],
                            errors="coerce",
                        ),
                    )

                    x[
                        "post_future_leakage"
                    ] = np.where(
                        x[
                            "controller_intervened"
                        ].eq(1),
                        0.0,
                        pd.to_numeric(
                            x[
                                ext_leak
                            ],
                            errors="coerce",
                        ),
                    )

                    x[
                        "false_control"
                    ] = (
                        x[
                            "task_type"
                        ]
                        .isin(
                            ANSWERABLE_TYPES
                        )
                        &
                        x[
                            "controller_intervened"
                        ].eq(1)
                    ).astype(int)

                    x[
                        "policy_failure"
                    ] = np.nan

                    c = (
                        x["task_type"]
                        .eq(
                            CLINICAL_TYPE
                        )
                    )

                    s = (
                        x["task_type"]
                        .eq(
                            SEQUENCE_TYPE
                        )
                    )

                    x.loc[
                        c,
                        "policy_failure",
                    ] = x.loc[
                        c,
                        "post_clinical_overreach",
                    ]

                    x.loc[
                        s,
                        "policy_failure",
                    ] = x.loc[
                        s,
                        "post_future_leakage",
                    ]

                    x[
                        "controller_variant"
                    ] = variant

                    ext_variants.append(
                        x
                    )

                ext_all = pd.concat(
                    ext_variants,
                    ignore_index=True,
                )

                ext_all.to_csv(
                    OUT
                    / "external_policy_controller_detail.csv",
                    index=False,
                )

                ext_summary_rows = []

                for variant, vg in ext_all.groupby(
                    "controller_variant"
                ):

                    a = vg[
                        vg[
                            "task_type"
                        ].isin(
                            ANSWERABLE_TYPES
                        )
                    ]

                    c = vg[
                        vg[
                            "task_type"
                        ].eq(
                            CLINICAL_TYPE
                        )
                    ]

                    s = vg[
                        vg[
                            "task_type"
                        ].eq(
                            SEQUENCE_TYPE
                        )
                    ]

                    p = vg[
                        vg[
                            "task_type"
                        ].isin(
                            POLICY_TYPES
                        )
                    ]

                    ext_summary_rows.append({
                        "variant":
                            variant,

                        "N":
                            len(vg),

                        "false_control_rate":
                            a[
                                "false_control"
                            ].mean(),

                        "clinical_overreach_rate":
                            c[
                                "post_clinical_overreach"
                            ].mean(),

                        "sequence_leakage_rate":
                            s[
                                "post_future_leakage"
                            ].mean(),

                        "overall_policy_failure_rate":
                            p[
                                "policy_failure"
                            ].mean(),

                        "policy_abstention_rate":
                            p[
                                "post_abstention"
                            ].mean(),
                    })

                external_summary_df = pd.DataFrame(
                    ext_summary_rows
                )

                external_summary_df.to_csv(
                    OUT
                    / "external_policy_controller_summary.csv",
                    index=False,
                )

                external_by_task_df = (
                    ext_all
                    .groupby(
                        [
                            "controller_variant",
                            "task_type",
                        ]
                    )
                    .agg(
                        N=(
                            "task_id",
                            "count",
                        ),
                        Intervention=(
                            "controller_intervened",
                            "mean",
                        ),
                        FalseControl=(
                            "false_control",
                            "mean",
                        ),
                        Abstention=(
                            "post_abstention",
                            "mean",
                        ),
                        ClinicalOverreach=(
                            "post_clinical_overreach",
                            "mean",
                        ),
                        FutureLeakage=(
                            "post_future_leakage",
                            "mean",
                        ),
                        PolicyFailure=(
                            "policy_failure",
                            "mean",
                        ),
                    )
                    .reset_index()
                )

                external_by_task_df.to_csv(
                    OUT
                    / "external_policy_controller_by_task.csv",
                    index=False,
                )

else:

    REPORT.append(
        "External trigger or external metrics file missing; external propagation skipped."
    )


# =============================================================================
# REPORT
# =============================================================================

section("3. PRIMARY CONTROLLER COMPARISON")

REPORT.append(
    summary_df.to_string(
        index=False
    )
)


section("4. PRIMARY CONTROLLER BY TASK FAMILY")

REPORT.append(
    by_task.to_string(
        index=False
    )
)


section("5. TASK-CLUSTERED 95% BOOTSTRAP CONFIDENCE INTERVALS")

REPORT.append(
    ci_df.to_string(
        index=False
    )
)


section("6. PAIRED TASK-CLUSTERED BOOTSTRAP TESTS")

REPORT.append(
    paired_df.to_string(
        index=False
    )
)

REPORT.append("")
REPORT.append(
    "Differences are challenger minus no-controller. "
    "For failure-rate metrics, negative values favor the controller."
)

REPORT.append(
    "Holm-adjusted p-values correct across the six controller-vs-baseline policy comparisons."
)


section("7. EXTERNAL LPM CONTROLLER COMPARISON")

if external_summary_df.empty:

    REPORT.append(
        "External output-level propagation was unavailable; use the previously computed external trigger metrics only."
    )

else:

    REPORT.append(
        external_summary_df.to_string(
            index=False
        )
    )

    REPORT.append("")
    REPORT.append(
        external_by_task_df.to_string(
            index=False
        )
    )


section("8. SCIENTIFIC INTERPRETATION")

REPORT.append(
    "The oracle-trigger controller is an explicit upper bound because it uses the benchmark task label only to decide whether policy control should activate."
)

REPORT.append(
    "The course-aware controller uses the previously generated out-of-fold trigger predictions. "
    "It therefore exposes both missed policy interventions and false interventions on answerable questions."
)

REPORT.append(
    "Neither controller receives a gold evidence identifier when producing its policy response."
)

REPORT.append(
    "No lexical answer F1 is reported for the policy-control comparison because deterministic abstention templates would mechanically inflate template-overlap scores."
)

REPORT.append(
    "Answer quality on answerable tasks remains the underlying Slide-indexed generation whenever the controller does not intervene; "
    "therefore controller cost is reported explicitly as the false-control / answer-preservation rate."
)

REPORT.append(
    "The course-aware controller should be presented as a deployable diagnostic baseline rather than as equivalent to the oracle upper bound if external transfer remains materially weaker."
)


report_text = "\n".join(
    REPORT
)

(
    OUT
    / "FINAL_POLICY_CONTROLLER_REPORT.txt"
).write_text(
    report_text,
    encoding="utf-8",
)

metadata = {
    "base_system":
        BASE_SYSTEM,

    "primary_outputs":
        len(base),

    "primary_tasks":
        int(
            base["task_id"]
            .nunique()
        ),

    "models":
        sorted(
            base[
                MODEL_COL
            ]
            .astype(str)
            .unique()
            .tolist()
        ),

    "bootstrap_iterations":
        BOOT,

    "bootstrap_unit":
        "task_id after averaging model-specific outputs within task",

    "course_aware_trigger_source":
        str(PRIMARY_TRIGGER),

    "external_trigger_source":
        str(EXTERNAL_TRIGGER),

    "gold_ids_used_by_policy_controller":
        False,

    "reference_answers_used_by_policy_controller":
        False,

    "policy_f1_reported":
        False,

    "clinical_response":
        CLINICAL_RESPONSE,

    "sequence_response":
        SEQUENCE_RESPONSE,
}

(
    OUT
    / "final_policy_controller_metadata.json"
).write_text(
    json.dumps(
        metadata,
        indent=2,
    ),
    encoding="utf-8",
)


print("")
print(report_text)

print("")
print("=" * 100)
print("OUTPUT FILES")
print("=" * 100)

for p in sorted(
    OUT.glob("*")
):
    print(p)

print("")
print(
    "FINAL POLICY-CONTROLLER ANALYSIS COMPLETE"
)
