#!/usr/bin/env python3

# Public path-portable copy of the frozen analysis source.
# Frozen analysis source SHA256: 58cd8e45be63d03801f95a8f89e974beab5895f792ebc28b727cac608b674a37

from pathlib import Path
import hashlib
import json
import math
import re

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[2]
INPUT = (
    ROOT
    / "artifacts/final_targeted_validation_20260822"
    / "position_aware_candidate_features.csv"
)
OUT = (
    ROOT
    / "artifacts/jiis_position_ablation_20260918"
)

SEED = 20260822
N_FOLDS = 5
N_BOOT = 100000

CONTENT_FEATURES = [
    "rrf_rr",
    "rrf_rank",
    "bm25_rr",
    "bm25_score",
    "tfidf_rr",
    "bge_sim",
]

POSITION_FEATURES = [
    "same_lecture",
    "lecture_distance",
]

FULL_FEATURES = CONTENT_FEATURES + POSITION_FEATURES

FORBIDDEN_FEATURES = [
    "signed_position_distance",
    "abs_position_distance",
    "slide_distance_same_lecture",
]

EXPECTED_ROWS = 29500
EXPECTED_TASKS = 600

# Frozen reproduction gates from the completed matched ablation.
EXPECTED_CONTENT_ANY3 = 178
EXPECTED_CONTENT_TARGET3 = 154

EXPECTED_FULL_ANY3 = 314
EXPECTED_FULL_TARGET3 = 269

EXPECTED_CONTENT_MRR = 0.271708745257992
EXPECTED_FULL_MRR = 0.4213431538126966



OUT.mkdir(parents=True, exist_ok=True)

def sha256(path):
    h = hashlib.sha256()

    with open(path, "rb") as f:
        for block in iter(
            lambda: f.read(1024 * 1024),
            b"",
        ):
            h.update(block)

    return h.hexdigest()


def make_lr():
    return Pipeline(
        [
            (
                "scale",
                StandardScaler(),
            ),
            (
                "logistic",
                LogisticRegression(
                    C=1.0,
                    max_iter=3000,
                    class_weight="balanced",
                    random_state=SEED,
                ),
            ),
        ]
    )


def evaluate_scored(scored, condition):
    rows = []

    scored = scored.copy()

    for task_id, group in scored.groupby(
        "task_id",
        sort=True,
    ):
        ranked = group.sort_values(
            ["score", "_input_order"],
            ascending=[False, True],
            kind="mergesort",
        )

        gold = ranked["is_gold"].to_numpy(int)
        target = ranked["is_target"].to_numpy(int)

        gold_pos = np.flatnonzero(gold == 1)
        target_pos = np.flatnonzero(target == 1)

        first_gold = (
            int(gold_pos[0] + 1)
            if len(gold_pos)
            else None
        )

        first_target = (
            int(target_pos[0] + 1)
            if len(target_pos)
            else None
        )

        rows.append(
            {
                "condition": condition,
                "task_id": str(task_id),
                "target_lecture": int(
                    ranked["target_lecture"].iloc[0]
                ),
                "any_gold_at_3": int(
                    ranked.iloc[:3]["is_gold"].any()
                ),
                "target_slide_at_3": int(
                    ranked.iloc[:3]["is_target"].any()
                ),
                "mrr_any_gold": (
                    1.0 / first_gold
                    if first_gold
                    else 0.0
                ),
                "mrr_target": (
                    1.0 / first_target
                    if first_target
                    else 0.0
                ),
                "first_gold_rank": first_gold,
                "first_target_rank": first_target,
            }
        )

    return pd.DataFrame(rows)


def cluster_bootstrap_delta(
    paired,
    diff_col,
    n_boot,
    seed,
):
    cluster = (
        paired.groupby(
            "target_lecture",
            as_index=False,
        )
        .agg(
            n_tasks=("task_id", "size"),
            diff_sum=(diff_col, "sum"),
            diff_mean=(diff_col, "mean"),
        )
    )

    cluster_sum = cluster[
        "diff_sum"
    ].to_numpy(float)

    cluster_n = cluster[
        "n_tasks"
    ].to_numpy(float)

    rng = np.random.default_rng(seed)

    c = len(cluster)
    values = np.empty(
        n_boot,
        dtype=float,
    )

    chunk = 5000
    done = 0

    while done < n_boot:

        m = min(
            chunk,
            n_boot - done,
        )

        idx = rng.integers(
            0,
            c,
            size=(m, c),
        )

        numerator = (
            cluster_sum[idx]
            .sum(axis=1)
        )

        denominator = (
            cluster_n[idx]
            .sum(axis=1)
        )

        values[
            done:done + m
        ] = numerator / denominator

        done += m

    ci_low, ci_high = np.quantile(
        values,
        [0.025, 0.975],
    )

    better = int(
        (
            cluster["diff_mean"]
            > 1e-15
        ).sum()
    )

    tied = int(
        np.isclose(
            cluster["diff_mean"],
            0.0,
            atol=1e-15,
        ).sum()
    )

    worse = int(
        (
            cluster["diff_mean"]
            < -1e-15
        ).sum()
    )

    return (
        float(ci_low),
        float(ci_high),
        better,
        tied,
        worse,
        cluster,
    )


print(
    "============================================================"
)
print(
    "LOAD FROZEN CANDIDATE MATRIX"
)
print(
    "============================================================"
)

df = pd.read_csv(INPUT)

df["task_id"] = (
    df["task_id"]
    .astype(str)
)

df["_input_order"] = np.arange(
    len(df)
)

if len(df) != EXPECTED_ROWS:
    raise RuntimeError(
        f"Expected {EXPECTED_ROWS} rows, "
        f"found {len(df)}"
    )

if (
    df["task_id"].nunique()
    != EXPECTED_TASKS
):
    raise RuntimeError(
        "Frozen task-count mismatch."
    )

missing = [
    x
    for x in FULL_FEATURES
    if x not in df.columns
]

if missing:
    raise RuntimeError(
        f"Missing features: {missing}"
    )

if (
    set(FULL_FEATURES)
    & set(FORBIDDEN_FEATURES)
):
    raise RuntimeError(
        "Forbidden exact-position feature "
        "entered full model."
    )


# ------------------------------------------------------------
# Recover candidate-document lecture IDs.
# ------------------------------------------------------------

doc_lecture = (
    df["doc"]
    .astype(str)
    .str.extract(
        r"lecture_(\d+)_slide_",
        expand=False,
    )
)

if doc_lecture.isna().any():
    bad = (
        df.loc[
            doc_lecture.isna(),
            "doc",
        ]
        .astype(str)
        .head(20)
        .tolist()
    )

    raise RuntimeError(
        "Could not parse lecture from "
        f"candidate doc IDs: {bad}"
    )

df["doc_lecture"] = (
    doc_lecture.astype(int)
)

lecture_values = sorted(
    int(x)
    for x in
    df["target_lecture"]
    .unique()
)

lecture_set = set(
    lecture_values
)

print("input:", INPUT)
print("sha256:", sha256(INPUT))
print("rows:", len(df))
print(
    "tasks:",
    df["task_id"].nunique(),
)
print(
    "target lectures:",
    lecture_values,
)
print(
    "number of target lectures:",
    len(lecture_values),
)


# ------------------------------------------------------------
# Reconstruct the exact grouped folds.
# ------------------------------------------------------------

print()
print(
    "============================================================"
)
print(
    "RECONSTRUCT TARGET-LECTURE-GROUPED FOLDS"
)
print(
    "============================================================"
)

task_ids = np.array(
    sorted(
        df["task_id"].unique()
    )
)

task_lecture = (
    df.groupby(
        "task_id",
        sort=True,
    )["target_lecture"]
    .first()
)

group_values = np.array(
    [
        task_lecture.loc[
            task_id
        ]
        for task_id in task_ids
    ]
)

cv = GroupKFold(
    n_splits=N_FOLDS
)

folds = list(
    cv.split(
        task_ids,
        groups=group_values,
    )
)

for fold_index, (
    train,
    test,
) in enumerate(
    folds,
    start=1,
):
    train_lectures = set(
        group_values[train]
    )

    test_lectures = set(
        group_values[test]
    )

    overlap = (
        train_lectures
        & test_lectures
    )

    print(
        f"fold {fold_index}: "
        f"train_tasks={len(train)} "
        f"test_tasks={len(test)} "
        f"overlap={len(overlap)}"
    )

    if overlap:
        raise RuntimeError(
            "Lecture leakage detected."
        )


# ------------------------------------------------------------
# Train on correct metadata.
# Test under:
#   correct metadata
#   true lecture - 1
#   true lecture + 1
#
# Candidate pool remains frozen.
# ------------------------------------------------------------

print()
print(
    "============================================================"
)
print(
    "TRAIN ON CORRECT METADATA; "
    "PERTURB TEST-TIME POSITION ONLY"
)
print(
    "============================================================"
)

correct_parts = []
content_parts = []

minus_parts = []
plus_parts = []

fold_model_rows = []

for fold_index, (
    train_index,
    test_index,
) in enumerate(
    folds,
    start=1,
):

    train_ids = set(
        task_ids[train_index]
    )

    test_ids = set(
        task_ids[test_index]
    )

    train_rows = (
        df[
            df["task_id"]
            .isin(train_ids)
        ]
        .copy()
    )

    test_rows = (
        df[
            df["task_id"]
            .isin(test_ids)
        ]
        .copy()
    )

    # ------------------------
    # Content-only model
    # ------------------------

    content_model = make_lr()

    content_model.fit(
        train_rows[
            CONTENT_FEATURES
        ].to_numpy(float),
        train_rows[
            "is_gold"
        ].to_numpy(int),
    )

    content_test = (
        test_rows.copy()
    )

    content_test["score"] = (
        content_model.predict_proba(
            content_test[
                CONTENT_FEATURES
            ].to_numpy(float)
        )[:, 1]
    )

    content_parts.append(
        content_test
    )

    # ------------------------
    # Full PACER model
    # trained on CORRECT metadata
    # ------------------------

    pacer_model = make_lr()

    pacer_model.fit(
        train_rows[
            FULL_FEATURES
        ].to_numpy(float),
        train_rows[
            "is_gold"
        ].to_numpy(int),
    )

    # Correct test metadata
    correct_test = (
        test_rows.copy()
    )

    correct_test["score"] = (
        pacer_model.predict_proba(
            correct_test[
                FULL_FEATURES
            ].to_numpy(float)
        )[:, 1]
    )

    correct_parts.append(
        correct_test
    )

    # ------------------------
    # One-lecture error
    # ------------------------

    for shift, collector in [
        (-1, minus_parts),
        (+1, plus_parts),
    ]:

        perturbed = (
            test_rows.copy()
        )

        perturbed[
            "assumed_lecture"
        ] = (
            perturbed[
                "target_lecture"
            ].astype(int)
            + shift
        )

        # Only include tasks for which the
        # shifted lecture actually exists.
        valid = (
            perturbed[
                "assumed_lecture"
            ]
            .isin(lecture_set)
        )

        perturbed = (
            perturbed[
                valid
            ]
            .copy()
        )

        perturbed[
            "same_lecture"
        ] = (
            perturbed[
                "doc_lecture"
            ].astype(int)
            == perturbed[
                "assumed_lecture"
            ].astype(int)
        ).astype(int)

        perturbed[
            "lecture_distance"
        ] = np.abs(
            perturbed[
                "doc_lecture"
            ].astype(int)
            - perturbed[
                "assumed_lecture"
            ].astype(int)
        )

        perturbed["score"] = (
            pacer_model.predict_proba(
                perturbed[
                    FULL_FEATURES
                ].to_numpy(float)
            )[:, 1]
        )

        collector.append(
            perturbed
        )

    fold_model_rows.append(
        {
            "fold": fold_index,
            "n_train_tasks": len(
                train_ids
            ),
            "n_test_tasks": len(
                test_ids
            ),
        }
    )

    print(
        f"fold {fold_index}: done"
    )


correct_scored = pd.concat(
    correct_parts,
    ignore_index=True,
)

content_scored = pd.concat(
    content_parts,
    ignore_index=True,
)

minus_scored = pd.concat(
    minus_parts,
    ignore_index=True,
)

plus_scored = pd.concat(
    plus_parts,
    ignore_index=True,
)


# ------------------------------------------------------------
# Convert scored candidates to per-task metrics.
# ------------------------------------------------------------

correct_metrics = evaluate_scored(
    correct_scored,
    "PACER correct metadata",
)

content_metrics = evaluate_scored(
    content_scored,
    "Content-only LR",
)

minus_metrics = evaluate_scored(
    minus_scored,
    "PACER assumed lecture -1",
)

plus_metrics = evaluate_scored(
    plus_scored,
    "PACER assumed lecture +1",
)

all_metrics = pd.concat(
    [
        content_metrics,
        correct_metrics,
        minus_metrics,
        plus_metrics,
    ],
    ignore_index=True,
)

all_metrics.to_csv(
    OUT
    / "metadata_stress_per_task_metrics.csv",
    index=False,
)


# ------------------------------------------------------------
# Frozen reproduction gate.
# ------------------------------------------------------------

print()
print(
    "============================================================"
)
print(
    "FROZEN REPRODUCTION GATES"
)
print(
    "============================================================"
)


def gate_counts(frame):
    return (
        int(
            frame[
                "any_gold_at_3"
            ].sum()
        ),
        int(
            frame[
                "target_slide_at_3"
            ].sum()
        ),
        float(
            frame[
                "mrr_any_gold"
            ].mean()
        ),
    )


content_any, (
    content_target
), content_mrr = gate_counts(
    content_metrics
)

full_any, (
    full_target
), full_mrr = gate_counts(
    correct_metrics
)

checks = [
    (
        "Content-only AnyGold@3",
        content_any,
        EXPECTED_CONTENT_ANY3,
    ),
    (
        "Content-only TargetSlide@3",
        content_target,
        EXPECTED_CONTENT_TARGET3,
    ),
    (
        "PACER AnyGold@3",
        full_any,
        EXPECTED_FULL_ANY3,
    ),
    (
        "PACER TargetSlide@3",
        full_target,
        EXPECTED_FULL_TARGET3,
    ),
]

passed = True

for name, observed, expected in checks:

    ok = (
        observed == expected
    )

    passed &= ok

    print(
        f"{name}: "
        f"observed={observed} "
        f"expected={expected} "
        f"{'PASS' if ok else 'FAIL'}"
    )

mrr_checks = [
    (
        "Content-only MRR",
        content_mrr,
        EXPECTED_CONTENT_MRR,
    ),
    (
        "PACER full MRR",
        full_mrr,
        EXPECTED_FULL_MRR,
    ),
]

for name, observed, expected in mrr_checks:

    ok = np.isclose(
        observed,
        expected,
        atol=1e-12,
        rtol=0,
    )

    passed &= bool(ok)

    print(
        f"{name}: "
        f"observed={observed:.12f} "
        f"expected={expected:.12f} "
        f"{'PASS' if ok else 'FAIL'}"
    )

if not passed:
    raise RuntimeError(
        "Frozen reproduction gate failed."
    )

print(
    "FROZEN REPRODUCTION: PASS"
)


# ------------------------------------------------------------
# Matched subset summaries for each perturbation.
# ------------------------------------------------------------

print()
print(
    "============================================================"
)
print(
    "MATCHED STRESS-TEST RESULTS"
)
print(
    "============================================================"
)

summary_rows = []
paired_rows = []
lecture_rows = []

metric_columns = {
    "AnyGold@3": (
        "any_gold_at_3"
    ),
    "TargetSlide@3": (
        "target_slide_at_3"
    ),
    "MRRAnyGold": (
        "mrr_any_gold"
    ),
}

shift_frames = {
    "-1": minus_metrics,
    "+1": plus_metrics,
}

for shift_index, (
    shift_name,
    perturbed_metrics,
) in enumerate(
    shift_frames.items()
):

    valid_ids = set(
        perturbed_metrics[
            "task_id"
        ]
    )

    correct_sub = (
        correct_metrics[
            correct_metrics[
                "task_id"
            ].isin(valid_ids)
        ]
        .copy()
    )

    content_sub = (
        content_metrics[
            content_metrics[
                "task_id"
            ].isin(valid_ids)
        ]
        .copy()
    )

    perturbed_sub = (
        perturbed_metrics.copy()
    )

    if not (
        len(correct_sub)
        == len(content_sub)
        == len(perturbed_sub)
    ):
        raise RuntimeError(
            "Matched subset size mismatch."
        )

    n = len(
        perturbed_sub
    )

    n_lectures = (
        perturbed_sub[
            "target_lecture"
        ].nunique()
    )

    print()
    print(
        "----------------------------------------"
    )
    print(
        f"ASSUMED CURRENT LECTURE {shift_name}"
    )
    print(
        "----------------------------------------"
    )
    print(
        f"matched tasks: {n}"
    )
    print(
        f"true target lectures: {n_lectures}"
    )

    for name, frame in [
        (
            "Content-only LR",
            content_sub,
        ),
        (
            "PACER correct metadata",
            correct_sub,
        ),
        (
            f"PACER assumed lecture {shift_name}",
            perturbed_sub,
        ),
    ]:

        row = {
            "shift": shift_name,
            "condition": name,
            "N": n,
            "n_true_lectures": n_lectures,
            "AnyGold@3": float(
                frame[
                    "any_gold_at_3"
                ].mean()
            ),
            "TargetSlide@3": float(
                frame[
                    "target_slide_at_3"
                ].mean()
            ),
            "MRRAnyGold": float(
                frame[
                    "mrr_any_gold"
                ].mean()
            ),
            "MedianFirstGoldRank": float(
                frame[
                    "first_gold_rank"
                ].median()
            ),
        }

        summary_rows.append(
            row
        )

        print(
            f"{name:28s} "
            f"Any@3={row['AnyGold@3']:.6f} "
            f"Target@3={row['TargetSlide@3']:.6f} "
            f"MRR={row['MRRAnyGold']:.6f} "
            f"MedianRank={row['MedianFirstGoldRank']:.1f}"
        )

    # ---------------------------------------
    # Paired comparisons
    # ---------------------------------------

    correct_p = (
        correct_sub
        .set_index(
            "task_id"
        )
    )

    content_p = (
        content_sub
        .set_index(
            "task_id"
        )
    )

    perturbed_p = (
        perturbed_sub
        .set_index(
            "task_id"
        )
    )

    common_ids = sorted(
        set(correct_p.index)
        & set(content_p.index)
        & set(perturbed_p.index)
    )

    for metric_index, (
        metric_name,
        metric_col,
    ) in enumerate(
        metric_columns.items()
    ):

        tmp = pd.DataFrame(
            {
                "task_id": common_ids,
                "target_lecture": [
                    int(
                        correct_p.loc[
                            tid,
                            "target_lecture",
                        ]
                    )
                    for tid in common_ids
                ],
                "correct": [
                    float(
                        correct_p.loc[
                            tid,
                            metric_col,
                        ]
                    )
                    for tid in common_ids
                ],
                "content": [
                    float(
                        content_p.loc[
                            tid,
                            metric_col,
                        ]
                    )
                    for tid in common_ids
                ],
                "perturbed": [
                    float(
                        perturbed_p.loc[
                            tid,
                            metric_col,
                        ]
                    )
                    for tid in common_ids
                ],
            }
        )

        tmp[
            "perturbed_minus_correct"
        ] = (
            tmp["perturbed"]
            - tmp["correct"]
        )

        tmp[
            "perturbed_minus_content"
        ] = (
            tmp["perturbed"]
            - tmp["content"]
        )

        # ------------------------------
        # Perturbed versus correct PACER
        # ------------------------------

        (
            ci_low,
            ci_high,
            lecture_better,
            lecture_tied,
            lecture_worse,
            cluster,
        ) = cluster_bootstrap_delta(
            tmp,
            "perturbed_minus_correct",
            N_BOOT,
            (
                SEED
                + 100 * shift_index
                + 10 * metric_index
                + 1
            ),
        )

        delta = float(
            tmp[
                "perturbed_minus_correct"
            ].mean()
        )

        paired_rows.append(
            {
                "shift": shift_name,
                "comparison":
                    "perturbed PACER - correct PACER",
                "metric": metric_name,
                "N": len(tmp),
                "n_true_lectures":
                    tmp[
                        "target_lecture"
                    ].nunique(),
                "reference_mean":
                    float(
                        tmp["correct"].mean()
                    ),
                "test_mean":
                    float(
                        tmp["perturbed"].mean()
                    ),
                "delta":
                    delta,
                "cluster_bootstrap_ci95_low":
                    ci_low,
                "cluster_bootstrap_ci95_high":
                    ci_high,
                "lectures_test_better":
                    lecture_better,
                "lectures_tied":
                    lecture_tied,
                "lectures_test_worse":
                    lecture_worse,
            }
        )

        for _, r in cluster.iterrows():
            lecture_rows.append(
                {
                    "shift": shift_name,
                    "comparison":
                        "perturbed PACER - correct PACER",
                    "metric": metric_name,
                    **r.to_dict(),
                }
            )

        # ------------------------------
        # Perturbed PACER vs content-only
        # ------------------------------

        (
            ci_low_c,
            ci_high_c,
            lecture_better_c,
            lecture_tied_c,
            lecture_worse_c,
            cluster_c,
        ) = cluster_bootstrap_delta(
            tmp,
            "perturbed_minus_content",
            N_BOOT,
            (
                SEED
                + 100 * shift_index
                + 10 * metric_index
                + 2
            ),
        )

        delta_c = float(
            tmp[
                "perturbed_minus_content"
            ].mean()
        )

        paired_rows.append(
            {
                "shift": shift_name,
                "comparison":
                    "perturbed PACER - content-only LR",
                "metric": metric_name,
                "N": len(tmp),
                "n_true_lectures":
                    tmp[
                        "target_lecture"
                    ].nunique(),
                "reference_mean":
                    float(
                        tmp["content"].mean()
                    ),
                "test_mean":
                    float(
                        tmp["perturbed"].mean()
                    ),
                "delta":
                    delta_c,
                "cluster_bootstrap_ci95_low":
                    ci_low_c,
                "cluster_bootstrap_ci95_high":
                    ci_high_c,
                "lectures_test_better":
                    lecture_better_c,
                "lectures_tied":
                    lecture_tied_c,
                "lectures_test_worse":
                    lecture_worse_c,
            }
        )

        for _, r in cluster_c.iterrows():
            lecture_rows.append(
                {
                    "shift": shift_name,
                    "comparison":
                        "perturbed PACER - content-only LR",
                    "metric": metric_name,
                    **r.to_dict(),
                }
            )


summary = pd.DataFrame(
    summary_rows
)

paired = pd.DataFrame(
    paired_rows
)

lecture_detail = pd.DataFrame(
    lecture_rows
)

summary.to_csv(
    OUT
    / "metadata_stress_summary.csv",
    index=False,
)

paired.to_csv(
    OUT
    / "metadata_stress_paired_deltas.csv",
    index=False,
)

lecture_detail.to_csv(
    OUT
    / "metadata_stress_per_lecture_deltas.csv",
    index=False,
)

pd.DataFrame(
    fold_model_rows
).to_csv(
    OUT
    / "metadata_stress_fold_audit.csv",
    index=False,
)


print()
print(
    "============================================================"
)
print(
    "PAIRED DEGRADATION / ROBUSTNESS"
)
print(
    "============================================================"
)

display_cols = [
    "shift",
    "comparison",
    "metric",
    "N",
    "reference_mean",
    "test_mean",
    "delta",
    "cluster_bootstrap_ci95_low",
    "cluster_bootstrap_ci95_high",
    "lectures_test_better",
    "lectures_tied",
    "lectures_test_worse",
]

print(
    paired[
        display_cols
    ].to_string(
        index=False
    )
)


# ------------------------------------------------------------
# Additional symmetric descriptive summary:
# for each task, average all VALID one-step-wrong
# directions available for that task.
#
# This is descriptive only and is NOT a single deployed
# perturbation condition.
# ------------------------------------------------------------

print()
print(
    "============================================================"
)
print(
    "DESCRIPTIVE SYMMETRIC ONE-STEP-ERROR SUMMARY"
)
print(
    "============================================================"
)

wrong = pd.concat(
    [
        minus_metrics.assign(
            direction="-1"
        ),
        plus_metrics.assign(
            direction="+1"
        ),
    ],
    ignore_index=True,
)

wrong_task = (
    wrong.groupby(
        [
            "task_id",
            "target_lecture",
        ],
        as_index=False,
    )
    .agg(
        n_valid_directions=(
            "direction",
            "nunique",
        ),
        AnyGold_at_3_wrong_mean=(
            "any_gold_at_3",
            "mean",
        ),
        TargetSlide_at_3_wrong_mean=(
            "target_slide_at_3",
            "mean",
        ),
        MRRAnyGold_wrong_mean=(
            "mrr_any_gold",
            "mean",
        ),
    )
)

correct_for_wrong = (
    correct_metrics[
        [
            "task_id",
            "target_lecture",
            "any_gold_at_3",
            "target_slide_at_3",
            "mrr_any_gold",
        ]
    ]
    .rename(
        columns={
            "any_gold_at_3":
                "AnyGold_at_3_correct",
            "target_slide_at_3":
                "TargetSlide_at_3_correct",
            "mrr_any_gold":
                "MRRAnyGold_correct",
        }
    )
)

wrong_task = (
    wrong_task.merge(
        correct_for_wrong,
        on=[
            "task_id",
            "target_lecture",
        ],
        how="left",
        validate="one_to_one",
    )
)

content_for_wrong = (
    content_metrics[
        [
            "task_id",
            "any_gold_at_3",
            "target_slide_at_3",
            "mrr_any_gold",
        ]
    ]
    .rename(
        columns={
            "any_gold_at_3":
                "AnyGold_at_3_content",
            "target_slide_at_3":
                "TargetSlide_at_3_content",
            "mrr_any_gold":
                "MRRAnyGold_content",
        }
    )
)

wrong_task = (
    wrong_task.merge(
        content_for_wrong,
        on="task_id",
        how="left",
        validate="one_to_one",
    )
)

wrong_task.to_csv(
    OUT
    / "metadata_stress_symmetric_task_summary.csv",
    index=False,
)

print(
    "tasks:",
    len(wrong_task),
)

print(
    "mean valid error directions/task:",
    wrong_task[
        "n_valid_directions"
    ].mean(),
)

print(
    "Correct PACER AnyGold@3:",
    wrong_task[
        "AnyGold_at_3_correct"
    ].mean(),
)

print(
    "Symmetric one-step wrong AnyGold@3:",
    wrong_task[
        "AnyGold_at_3_wrong_mean"
    ].mean(),
)

print(
    "Content-only AnyGold@3:",
    wrong_task[
        "AnyGold_at_3_content"
    ].mean(),
)

print(
    "Correct PACER TargetSlide@3:",
    wrong_task[
        "TargetSlide_at_3_correct"
    ].mean(),
)

print(
    "Symmetric one-step wrong TargetSlide@3:",
    wrong_task[
        "TargetSlide_at_3_wrong_mean"
    ].mean(),
)

print(
    "Content-only TargetSlide@3:",
    wrong_task[
        "TargetSlide_at_3_content"
    ].mean(),
)

print(
    "Correct PACER MRR:",
    wrong_task[
        "MRRAnyGold_correct"
    ].mean(),
)

print(
    "Symmetric one-step wrong MRR:",
    wrong_task[
        "MRRAnyGold_wrong_mean"
    ].mean(),
)

print(
    "Content-only MRR:",
    wrong_task[
        "MRRAnyGold_content"
    ].mean(),
)


# ------------------------------------------------------------
# Metadata / report
# ------------------------------------------------------------

metadata = {
    "analysis":
        "one-lecture current-position metadata-error stress test",
    "input":
        str(INPUT),
    "input_sha256":
        sha256(INPUT),
    "candidate_pool":
        "frozen position-constrained RRF top-50 candidate pool",
    "training_metadata":
        "correct target/current lecture",
    "test_perturbations": [
        "assumed current lecture = true lecture - 1",
        "assumed current lecture = true lecture + 1",
    ],
    "recomputed_test_features": [
        "same_lecture",
        "lecture_distance",
    ],
    "unchanged_features":
        CONTENT_FEATURES,
    "forbidden_features":
        FORBIDDEN_FEATURES,
    "important_scope_limitation":
        (
            "The candidate pool is held fixed from the correctly "
            "position-constrained retrieval stage. Therefore this "
            "experiment measures reranking sensitivity to current-lecture "
            "metadata error, not end-to-end errors in candidate eligibility "
            "or retrieval caused by a misidentified learner position."
        ),
    "seed":
        SEED,
    "cluster_bootstrap_replicates":
        N_BOOT,
}

(
    OUT
    / "METADATA_STRESS_TEST.json"
).write_text(
    json.dumps(
        metadata,
        indent=2,
    )
    + "\n",
    encoding="utf-8",
)


report = []

report.append(
    "# PACER One-Lecture Metadata-Error Stress Test"
)

report.append("")
report.append(
    "PACER is trained under correct course-position metadata. "
    "At test time only, the assumed current lecture is shifted "
    "by one lecture backward or forward, and `same_lecture` and "
    "`lecture_distance` are recomputed. Content features and the "
    "frozen candidate pool are unchanged."
)

report.append("")
report.append(
    "Important scope: this is a reranking sensitivity analysis. "
    "It does not simulate errors in the upstream eligibility mask "
    "or candidate-generation stage."
)

report.append("")
report.append(
    "## Matched condition summaries"
)
report.append("")
report.append(
    summary.to_csv(
        index=False
    )
)

report.append("")
report.append(
    "## Paired cluster-bootstrap comparisons"
)
report.append("")
report.append(
    paired.to_csv(
        index=False
    )
)

(
    OUT
    / "METADATA_STRESS_RESULTS.md"
).write_text(
    "\n".join(report),
    encoding="utf-8",
)


print()
print(
    "============================================================"
)
print(
    "OUTPUTS"
)
print(
    "============================================================"
)

for filename in [
    "metadata_stress_summary.csv",
    "metadata_stress_paired_deltas.csv",
    "metadata_stress_per_lecture_deltas.csv",
    "metadata_stress_per_task_metrics.csv",
    "metadata_stress_symmetric_task_summary.csv",
    "metadata_stress_fold_audit.csv",
    "METADATA_STRESS_TEST.json",
    "METADATA_STRESS_RESULTS.md",
]:
    print(
        OUT / filename
    )

print()
print(
    "============================================================"
)
print(
    "STRESS TEST COMPLETE"
)
print(
    "============================================================"
)
