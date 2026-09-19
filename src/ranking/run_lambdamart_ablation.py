#!/usr/bin/env python3

# Public path-portable copy of the frozen analysis source.
# Frozen analysis source SHA256: cb87f619eeefebe4ac9d29d5a569a8ac714858acd85e61b773e1c7ce50357e9a

from pathlib import Path
import hashlib
import itertools
import json
import math
import platform

import numpy as np
import pandas as pd
import sklearn
import lightgbm as lgb

from lightgbm import LGBMRanker
from sklearn.model_selection import GroupKFold


ROOT = Path(__file__).resolve().parents[2]

INPUT = (
    ROOT
    / "artifacts/final_targeted_validation_20260822"
    / "position_aware_candidate_features.csv"
)

OLD = (
    ROOT
    / "artifacts/jiis_position_ablation_20260918"
)

OUT = (
    ROOT
    / "artifacts/jiis_lambdamart_reliability_20260918"
)

SEED = 20260822

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

FULL_FEATURES = (
    CONTENT_FEATURES
    + POSITION_FEATURES
)

FORBIDDEN = [
    "signed_position_distance",
    "abs_position_distance",
    "slide_distance_same_lecture",
]

EXPECTED_ROWS = 29500
EXPECTED_TASKS = 600

N_OUTER = 5
N_INNER = 3
N_BOOT = 100000


# A deliberately small, preregistered grid.
# Selection occurs using ONLY outer-training lectures.
PARAM_GRID = [
    {
        "num_leaves": leaves,
        "min_child_samples": child,
        "n_estimators": estimators,
        "learning_rate": 0.05,
    }
    for leaves, child, estimators
    in itertools.product(
        [7, 15, 31],
        [10, 30],
        [150, 300],
    )
]



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


def prepare_ranker_rows(
    frame,
    features,
):
    x = (
        frame
        .sort_values(
            ["task_id", "_input_order"],
            kind="mergesort",
        )
        .copy()
    )

    groups = (
        x.groupby(
            "task_id",
            sort=False,
        )
        .size()
        .to_numpy(int)
    )

    X = x[
        features
    ].to_numpy(float)

    y = x[
        "is_gold"
    ].to_numpy(int)

    return x, X, y, groups


def make_ranker(params):
    return LGBMRanker(
        objective="lambdarank",
        boosting_type="gbdt",
        importance_type="gain",
        random_state=SEED,
        deterministic=True,
        force_col_wise=True,
        verbosity=-1,
        reg_lambda=1.0,
        reg_alpha=0.0,
        subsample=1.0,
        colsample_bytree=1.0,
        max_depth=-1,
        lambdarank_truncation_level=10,
        **params,
    )


def fit_ranker(
    train_rows,
    features,
    params,
):
    ordered, X, y, groups = (
        prepare_ranker_rows(
            train_rows,
            features,
        )
    )

    model = make_ranker(
        params
    )

    model.fit(
        X,
        y,
        group=groups,
    )

    return model


def score_rows(
    model,
    rows,
    features,
):
    out = rows.copy()

    out["score"] = model.predict(
        out[
            features
        ].to_numpy(float)
    )

    return out


def per_task_metrics(
    scored,
    method,
):
    rows = []

    for task_id, g in scored.groupby(
        "task_id",
        sort=True,
    ):
        ranked = g.sort_values(
            ["score", "_input_order"],
            ascending=[False, True],
            kind="mergesort",
        )

        gold = ranked[
            "is_gold"
        ].to_numpy(int)

        target = ranked[
            "is_target"
        ].to_numpy(int)

        gold_idx = np.flatnonzero(
            gold == 1
        )

        target_idx = np.flatnonzero(
            target == 1
        )

        first_gold = (
            int(gold_idx[0] + 1)
            if len(gold_idx)
            else None
        )

        first_target = (
            int(target_idx[0] + 1)
            if len(target_idx)
            else None
        )

        rows.append(
            {
                "method": method,
                "task_id": str(task_id),
                "target_lecture": int(
                    ranked[
                        "target_lecture"
                    ].iloc[0]
                ),
                "AnyGold@1": int(
                    ranked.iloc[:1][
                        "is_gold"
                    ].any()
                ),
                "AnyGold@3": int(
                    ranked.iloc[:3][
                        "is_gold"
                    ].any()
                ),
                "AnyGold@5": int(
                    ranked.iloc[:5][
                        "is_gold"
                    ].any()
                ),
                "AnyGold@10": int(
                    ranked.iloc[:10][
                        "is_gold"
                    ].any()
                ),
                "TargetSlide@3": int(
                    ranked.iloc[:3][
                        "is_target"
                    ].any()
                ),
                "MRRAnyGold": (
                    1.0 / first_gold
                    if first_gold
                    else 0.0
                ),
                "MRRTargetSlide": (
                    1.0 / first_target
                    if first_target
                    else 0.0
                ),
                "FirstGoldRank": (
                    first_gold
                    if first_gold
                    else np.nan
                ),
            }
        )

    return pd.DataFrame(
        rows
    )


def summarize(metrics):
    return {
        "N": len(metrics),
        "AnyGold_hits": int(
            metrics[
                "AnyGold@3"
            ].sum()
        ),
        "AnyGold@3": float(
            metrics[
                "AnyGold@3"
            ].mean()
        ),
        "TargetSlide_hits": int(
            metrics[
                "TargetSlide@3"
            ].sum()
        ),
        "TargetSlide@3": float(
            metrics[
                "TargetSlide@3"
            ].mean()
        ),
        "MRRAnyGold": float(
            metrics[
                "MRRAnyGold"
            ].mean()
        ),
        "MRRTargetSlide": float(
            metrics[
                "MRRTargetSlide"
            ].mean()
        ),
        "MedianFirstGoldRank": float(
            metrics[
                "FirstGoldRank"
            ].median()
        ),
    }


def cluster_bootstrap_delta(
    merged,
    diff_col,
    n_boot,
    seed,
):
    by_lecture = (
        merged.groupby(
            "target_lecture",
            as_index=False,
        )
        .agg(
            n_tasks=("task_id", "size"),
            diff_sum=(diff_col, "sum"),
            diff_mean=(diff_col, "mean"),
        )
        .sort_values(
            "target_lecture"
        )
    )

    sums = (
        by_lecture[
            "diff_sum"
        ].to_numpy(float)
    )

    ns = (
        by_lecture[
            "n_tasks"
        ].to_numpy(float)
    )

    rng = np.random.default_rng(
        seed
    )

    c = len(
        by_lecture
    )

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

        values[
            done:done + m
        ] = (
            sums[idx].sum(axis=1)
            / ns[idx].sum(axis=1)
        )

        done += m

    ci = np.quantile(
        values,
        [0.025, 0.975],
    )

    improved = int(
        (
            by_lecture[
                "diff_mean"
            ] > 1e-15
        ).sum()
    )

    tied = int(
        np.isclose(
            by_lecture[
                "diff_mean"
            ],
            0.0,
            atol=1e-15,
        ).sum()
    )

    worse = int(
        (
            by_lecture[
                "diff_mean"
            ] < -1e-15
        ).sum()
    )

    return (
        float(ci[0]),
        float(ci[1]),
        improved,
        tied,
        worse,
    )


def evaluate_parameter_set(
    train_frame,
    features,
    params,
    inner_folds,
    inner_task_ids,
):
    vals = []

    for train_idx, val_idx in inner_folds:

        fit_ids = set(
            inner_task_ids[
                train_idx
            ]
        )

        val_ids = set(
            inner_task_ids[
                val_idx
            ]
        )

        fit_rows = train_frame[
            train_frame[
                "task_id"
            ].isin(
                fit_ids
            )
        ].copy()

        val_rows = train_frame[
            train_frame[
                "task_id"
            ].isin(
                val_ids
            )
        ].copy()

        model = fit_ranker(
            fit_rows,
            features,
            params,
        )

        scored = score_rows(
            model,
            val_rows,
            features,
        )

        metrics = per_task_metrics(
            scored,
            "inner",
        )

        vals.append(
            {
                "AnyGold@3": float(
                    metrics[
                        "AnyGold@3"
                    ].mean()
                ),
                "MRRAnyGold": float(
                    metrics[
                        "MRRAnyGold"
                    ].mean()
                ),
            }
        )

    return {
        "mean_AnyGold@3": float(
            np.mean(
                [
                    x["AnyGold@3"]
                    for x in vals
                ]
            )
        ),
        "mean_MRRAnyGold": float(
            np.mean(
                [
                    x["MRRAnyGold"]
                    for x in vals
                ]
            )
        ),
    }


print(
    "============================================================"
)
print(
    "LOAD AND VALIDATE FROZEN CANDIDATE MATRIX"
)
print(
    "============================================================"
)

df = pd.read_csv(
    INPUT
)

df["task_id"] = (
    df["task_id"]
    .astype(str)
)

df["_input_order"] = np.arange(
    len(df)
)

print(
    "input:",
    INPUT,
)

print(
    "sha256:",
    sha256(INPUT),
)

print(
    "rows:",
    len(df),
)

print(
    "tasks:",
    df[
        "task_id"
    ].nunique(),
)

print(
    "lectures:",
    df[
        "target_lecture"
    ].nunique(),
)

print(
    "lightgbm:",
    lgb.__version__,
)

if len(df) != EXPECTED_ROWS:
    raise RuntimeError(
        "Frozen row count mismatch."
    )

if (
    df["task_id"].nunique()
    != EXPECTED_TASKS
):
    raise RuntimeError(
        "Frozen task count mismatch."
    )

for f in FULL_FEATURES:
    if f not in df.columns:
        raise RuntimeError(
            f"Missing feature: {f}"
        )

if (
    set(FULL_FEATURES)
    & set(FORBIDDEN)
):
    raise RuntimeError(
        "Forbidden exact-position feature entered model."
    )


# ------------------------------------------------------------
# Outer grouped CV
# ------------------------------------------------------------

task_ids = np.array(
    sorted(
        df[
            "task_id"
        ].unique()
    )
)

task_lecture = (
    df.groupby(
        "task_id",
        sort=True,
    )[
        "target_lecture"
    ]
    .first()
)

groups = np.array(
    [
        task_lecture.loc[t]
        for t in task_ids
    ]
)

outer_cv = GroupKFold(
    n_splits=N_OUTER
)

outer_folds = list(
    outer_cv.split(
        task_ids,
        groups=groups,
    )
)

print()
print(
    "============================================================"
)
print(
    "OUTER 5-FOLD TARGET-LECTURE CV"
)
print(
    "============================================================"
)

for fold, (
    train_idx,
    test_idx,
) in enumerate(
    outer_folds,
    start=1,
):
    train_lectures = set(
        groups[
            train_idx
        ]
    )

    test_lectures = set(
        groups[
            test_idx
        ]
    )

    print(
        f"fold {fold}: "
        f"train_tasks={len(train_idx)} "
        f"test_tasks={len(test_idx)} "
        f"train_lectures={len(train_lectures)} "
        f"test_lectures={len(test_lectures)} "
        f"overlap={len(train_lectures & test_lectures)}"
    )

    if (
        train_lectures
        & test_lectures
    ):
        raise RuntimeError(
            "Outer lecture leakage."
        )


feature_sets = {
    "Content-only LambdaMART":
        CONTENT_FEATURES,
    "Full PACER LambdaMART":
        FULL_FEATURES,
}

scored_parts = {
    name: []
    for name in feature_sets
}

tuning_rows = []


print()
print(
    "============================================================"
)
print(
    "NESTED HYPERPARAMETER SELECTION"
)
print(
    "============================================================"
)

for outer_fold, (
    outer_train_idx,
    outer_test_idx,
) in enumerate(
    outer_folds,
    start=1,
):

    outer_train_ids = (
        task_ids[
            outer_train_idx
        ]
    )

    outer_test_ids = (
        task_ids[
            outer_test_idx
        ]
    )

    outer_train_rows = (
        df[
            df["task_id"].isin(
                set(
                    outer_train_ids
                )
            )
        ]
        .copy()
    )

    outer_test_rows = (
        df[
            df["task_id"].isin(
                set(
                    outer_test_ids
                )
            )
        ]
        .copy()
    )

    inner_groups = np.array(
        [
            task_lecture.loc[t]
            for t in outer_train_ids
        ]
    )

    inner_cv = GroupKFold(
        n_splits=N_INNER
    )

    inner_folds = list(
        inner_cv.split(
            outer_train_ids,
            groups=inner_groups,
        )
    )

    print()
    print(
        f"----- OUTER FOLD {outer_fold} -----"
    )

    for method, features in (
        feature_sets.items()
    ):

        print(
            f"{method}: tuning "
            f"{len(PARAM_GRID)} configs"
        )

        candidates = []

        for param_index, params in enumerate(
            PARAM_GRID,
            start=1,
        ):
            scores = (
                evaluate_parameter_set(
                    outer_train_rows,
                    features,
                    params,
                    inner_folds,
                    outer_train_ids,
                )
            )

            row = {
                "outer_fold":
                    outer_fold,
                "method":
                    method,
                "param_index":
                    param_index,
                **params,
                **scores,
            }

            tuning_rows.append(
                row
            )

            candidates.append(
                row
            )

        # Primary selection:
        # highest mean AnyGold@3.
        # Tie-break:
        # highest MRR.
        # Then simpler model.
        candidates = sorted(
            candidates,
            key=lambda x: (
                -x[
                    "mean_AnyGold@3"
                ],
                -x[
                    "mean_MRRAnyGold"
                ],
                x[
                    "num_leaves"
                ],
                x[
                    "n_estimators"
                ],
                x[
                    "min_child_samples"
                ],
            ),
        )

        best = candidates[0]

        best_params = {
            "num_leaves":
                int(
                    best[
                        "num_leaves"
                    ]
                ),
            "min_child_samples":
                int(
                    best[
                        "min_child_samples"
                    ]
                ),
            "n_estimators":
                int(
                    best[
                        "n_estimators"
                    ]
                ),
            "learning_rate":
                float(
                    best[
                        "learning_rate"
                    ]
                ),
        }

        print(
            "  selected:",
            best_params,
            "inner Any@3=",
            f"{best['mean_AnyGold@3']:.6f}",
            "inner MRR=",
            f"{best['mean_MRRAnyGold']:.6f}",
        )

        final_model = fit_ranker(
            outer_train_rows,
            features,
            best_params,
        )

        scored = score_rows(
            final_model,
            outer_test_rows,
            features,
        )

        scored[
            "outer_fold"
        ] = outer_fold

        scored_parts[
            method
        ].append(
            scored
        )


tuning_df = pd.DataFrame(
    tuning_rows
)

tuning_df.to_csv(
    OUT
    / "lambdamart_nested_tuning.csv",
    index=False,
)


print()
print(
    "============================================================"
)
print(
    "OUTER-FOLD HELD-OUT RESULTS"
)
print(
    "============================================================"
)

all_metrics = []
summary_rows = []

for method in feature_sets:

    scored = pd.concat(
        scored_parts[
            method
        ],
        ignore_index=True,
    )

    scored.to_csv(
        OUT
        / (
            method
            .lower()
            .replace(" ", "_")
            .replace("-", "_")
            + "_scored_candidates.csv"
        ),
        index=False,
    )

    metrics = per_task_metrics(
        scored,
        method,
    )

    all_metrics.append(
        metrics
    )

    s = summarize(
        metrics
    )

    summary_rows.append(
        {
            "method": method,
            **s,
        }
    )

    print()
    print(method)

    for k, v in s.items():
        print(
            f"  {k}: {v}"
        )


metrics_df = pd.concat(
    all_metrics,
    ignore_index=True,
)

metrics_df.to_csv(
    OUT
    / "lambdamart_per_task_metrics.csv",
    index=False,
)

summary_df = pd.DataFrame(
    summary_rows
)

summary_df.to_csv(
    OUT
    / "lambdamart_summary.csv",
    index=False,
)


# ------------------------------------------------------------
# Paired content vs full LambdaMART
# ------------------------------------------------------------

print()
print(
    "============================================================"
)
print(
    "PAIRED FULL LAMBDAMART VS CONTENT-ONLY LAMBDAMART"
)
print(
    "============================================================"
)

content = (
    metrics_df[
        metrics_df["method"]
        == "Content-only LambdaMART"
    ]
    .set_index(
        "task_id"
    )
)

full = (
    metrics_df[
        metrics_df["method"]
        == "Full PACER LambdaMART"
    ]
    .set_index(
        "task_id"
    )
)

common = sorted(
    set(content.index)
    & set(full.index)
)

comparison_rows = []

for idx, metric in enumerate(
    [
        "AnyGold@3",
        "TargetSlide@3",
        "MRRAnyGold",
    ]
):

    tmp = pd.DataFrame(
        {
            "task_id": common,
            "target_lecture": [
                int(
                    full.loc[
                        t,
                        "target_lecture",
                    ]
                )
                for t in common
            ],
            "content": [
                float(
                    content.loc[
                        t,
                        metric,
                    ]
                )
                for t in common
            ],
            "full": [
                float(
                    full.loc[
                        t,
                        metric,
                    ]
                )
                for t in common
            ],
        }
    )

    tmp["diff"] = (
        tmp["full"]
        - tmp["content"]
    )

    delta = float(
        tmp["diff"].mean()
    )

    (
        ci_low,
        ci_high,
        improved,
        tied,
        worse,
    ) = cluster_bootstrap_delta(
        tmp,
        "diff",
        N_BOOT,
        SEED + idx + 100,
    )

    row = {
        "comparison":
            "Full PACER LambdaMART - Content-only LambdaMART",
        "metric": metric,
        "N": len(tmp),
        "n_lectures":
            tmp[
                "target_lecture"
            ].nunique(),
        "content_mean":
            float(
                tmp[
                    "content"
                ].mean()
            ),
        "full_mean":
            float(
                tmp[
                    "full"
                ].mean()
            ),
        "delta":
            delta,
        "cluster_ci95_low":
            ci_low,
        "cluster_ci95_high":
            ci_high,
        "lectures_improved":
            improved,
        "lectures_tied":
            tied,
        "lectures_worse":
            worse,
    }

    comparison_rows.append(
        row
    )

    print()
    print(metric)
    print(
        f"  content = "
        f"{row['content_mean']:.6f}"
    )
    print(
        f"  full    = "
        f"{row['full_mean']:.6f}"
    )
    print(
        f"  delta   = "
        f"{row['delta']:+.6f}"
    )
    print(
        "  cluster bootstrap 95% CI = "
        f"[{ci_low:+.6f}, "
        f"{ci_high:+.6f}]"
    )
    print(
        "  lectures improved/tied/worse = "
        f"{improved}/{tied}/{worse}"
    )


comparison_df = pd.DataFrame(
    comparison_rows
)

comparison_df.to_csv(
    OUT
    / "lambdamart_full_vs_content_cluster.csv",
    index=False,
)


# ------------------------------------------------------------
# Compare against previously frozen LR results if available.
# ------------------------------------------------------------

print()
print(
    "============================================================"
)
print(
    "CONTEXT WITH PREVIOUS LR RESULTS"
)
print(
    "============================================================"
)

old_summary_path = (
    OLD
    / "ablation_summary_at3.csv"
)

if old_summary_path.exists():
    old = pd.read_csv(
        old_summary_path
    )

    print(
        old.to_string(
            index=False
        )
    )

print()
print(
    summary_df.to_string(
        index=False
    )
)


metadata = {
    "experiment":
        "Nested-CV LambdaMART feature-family ablation",
    "input":
        str(INPUT),
    "input_sha256":
        sha256(INPUT),
    "rows":
        len(df),
    "tasks":
        int(
            df[
                "task_id"
            ].nunique()
        ),
    "outer_cv":
        "5-fold GroupKFold by target_lecture",
    "inner_cv":
        "3-fold GroupKFold by target_lecture within each outer-training split",
    "selection_metric":
        "mean inner AnyGold@3, then mean MRRAnyGold",
    "content_features":
        CONTENT_FEATURES,
    "position_features":
        POSITION_FEATURES,
    "full_features":
        FULL_FEATURES,
    "forbidden_exact_position_features":
        FORBIDDEN,
    "parameter_grid":
        PARAM_GRID,
    "seed":
        SEED,
    "python":
        platform.python_version(),
    "numpy":
        np.__version__,
    "pandas":
        pd.__version__,
    "sklearn":
        sklearn.__version__,
    "lightgbm":
        lgb.__version__,
}

(
    OUT
    / "LAMBDAMART_METADATA.json"
).write_text(
    json.dumps(
        metadata,
        indent=2,
    )
    + "\n",
    encoding="utf-8",
)

print()
print(
    "============================================================"
)
print(
    "LAMBDAMART EXPERIMENT COMPLETE"
)
print(
    "============================================================"
)
