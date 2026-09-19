#!/usr/bin/env python3

# Public path-portable copy of the frozen analysis source.
# Frozen analysis source SHA256: eca2e04c6a4d2b617d4a8e477457318f7096316fc6bf3e607cfece3763f6b33b

from pathlib import Path
import json
import math

import numpy as np
import pandas as pd
from lightgbm import LGBMRanker
from sklearn.model_selection import GroupKFold


ROOT = Path(__file__).resolve().parents[2]

BASE = (
    ROOT
    / "artifacts/jiis_lambdamart_reliability_20260918"
)

OUT = (
    ROOT
    / "artifacts/jiis_final_experiments_20260918"
)

CANDIDATES = (
    ROOT
    / "artifacts/final_targeted_validation_20260822"
    / "position_aware_candidate_features.csv"
)

TUNING = BASE / "lambdamart_nested_tuning.csv"
OLD_METRICS = BASE / "lambdamart_per_task_metrics.csv"

SEED = 20260822
N_BOOT = 100000

CONTENT_FEATURES = [
    "rrf_rr",
    "rrf_rank",
    "bm25_rr",
    "bm25_score",
    "tfidf_rr",
    "bge_sim",
]

FULL_FEATURES = CONTENT_FEATURES + [
    "same_lecture",
    "lecture_distance",
]



OUT.mkdir(parents=True, exist_ok=True)

def find_candidate_lecture_column(df):
    aliases = [
        "candidate_lecture",
        "candidate_lecture_id",
        "doc_lecture",
        "doc_lecture_id",
        "lecture_id",
        "lecture",
    ]

    valid = []

    for c in aliases:
        if c not in df.columns:
            continue

        try:
            cand = pd.to_numeric(
                df[c],
                errors="raise",
            ).to_numpy()

            target = pd.to_numeric(
                df["target_lecture"],
                errors="raise",
            ).to_numpy()

            expected_same = (
                cand == target
            ).astype(int)

            expected_dist = np.abs(
                cand - target
            )

            same_ok = np.array_equal(
                expected_same,
                df["same_lecture"]
                .to_numpy(int),
            )

            dist_ok = np.allclose(
                expected_dist,
                df["lecture_distance"]
                .to_numpy(float),
                atol=1e-12,
                rtol=0,
            )

            if same_ok and dist_ok:
                valid.append(c)

        except Exception:
            pass

    if len(valid) != 1:
        raise RuntimeError(
            "Could not uniquely identify candidate lecture column. "
            f"Consistent candidates={valid}; columns={list(df.columns)}"
        )

    return valid[0]


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
        num_leaves=int(params["num_leaves"]),
        min_child_samples=int(
            params["min_child_samples"]
        ),
        n_estimators=int(
            params["n_estimators"]
        ),
        learning_rate=float(
            params["learning_rate"]
        ),
    )


def fit_ranker(frame, features, params):
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

    model = make_ranker(params)

    # Keep DataFrame feature names both at fit and predict.
    model.fit(
        x[features],
        x["is_gold"].to_numpy(int),
        group=groups,
    )

    return model


def score(model, frame, features):
    out = frame.copy()

    out["score"] = model.predict(
        out[features]
    )

    return out


def per_task(scored, condition):
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

        gold = np.flatnonzero(
            ranked["is_gold"].to_numpy(int)
            == 1
        )

        target = np.flatnonzero(
            ranked["is_target"].to_numpy(int)
            == 1
        )

        fg = (
            int(gold[0] + 1)
            if len(gold)
            else None
        )

        ft = (
            int(target[0] + 1)
            if len(target)
            else None
        )

        rows.append(
            {
                "condition": condition,
                "task_id": str(task_id),
                "target_lecture": int(
                    ranked[
                        "target_lecture"
                    ].iloc[0]
                ),
                "AnyGold@3": int(
                    ranked.iloc[:3][
                        "is_gold"
                    ].any()
                ),
                "TargetSlide@3": int(
                    ranked.iloc[:3][
                        "is_target"
                    ].any()
                ),
                "MRRAnyGold": (
                    1.0 / fg
                    if fg
                    else 0.0
                ),
                "MRRTargetSlide": (
                    1.0 / ft
                    if ft
                    else 0.0
                ),
                "FirstGoldRank": (
                    fg
                    if fg
                    else np.nan
                ),
            }
        )

    return pd.DataFrame(rows)


def summarize(x):
    return {
        "N": len(x),
        "lectures": int(
            x["target_lecture"].nunique()
        ),
        "AnyGold@3": float(
            x["AnyGold@3"].mean()
        ),
        "TargetSlide@3": float(
            x["TargetSlide@3"].mean()
        ),
        "MRRAnyGold": float(
            x["MRRAnyGold"].mean()
        ),
        "MedianFirstGoldRank": float(
            x["FirstGoldRank"].median()
        ),
    }


def cluster_bootstrap_delta(
    df,
    a_col,
    b_col,
    n_boot,
    seed,
):
    tmp = df.copy()

    tmp["diff"] = (
        tmp[a_col]
        - tmp[b_col]
    )

    cluster = (
        tmp.groupby(
            "target_lecture",
            as_index=False,
        )
        .agg(
            n=("task_id", "size"),
            diff_sum=("diff", "sum"),
        )
    )

    n = cluster["n"].to_numpy(float)
    s = cluster[
        "diff_sum"
    ].to_numpy(float)

    rng = np.random.default_rng(seed)

    B = np.empty(
        n_boot,
        dtype=float,
    )

    c = len(cluster)
    done = 0

    while done < n_boot:
        m = min(
            5000,
            n_boot - done,
        )

        ix = rng.integers(
            0,
            c,
            size=(m, c),
        )

        B[
            done:done + m
        ] = (
            s[ix].sum(axis=1)
            / n[ix].sum(axis=1)
        )

        done += m

    return tuple(
        np.quantile(
            B,
            [0.025, 0.975],
        )
    )


def break_even(correct, wrong, content):
    den = correct - wrong

    if abs(den) < 1e-12:
        return np.nan

    return (
        content - wrong
    ) / den


def cluster_bootstrap_break_even(
    frame,
    correct_col,
    wrong_col,
    content_col,
    seed,
):
    agg = (
        frame.groupby(
            "target_lecture",
            as_index=False,
        )
        .agg(
            n=("task_id", "size"),
            c=(correct_col, "sum"),
            w=(wrong_col, "sum"),
            b=(content_col, "sum"),
        )
    )

    arr = agg[
        ["n", "c", "w", "b"]
    ].to_numpy(float)

    rng = np.random.default_rng(seed)
    C = len(arr)

    vals = []
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

        z = arr[ix].sum(axis=1)

        n = z[:, 0]
        c = z[:, 1] / n
        w = z[:, 2] / n
        b = z[:, 3] / n

        den = c - w

        p = np.full(
            m,
            np.nan,
            dtype=float,
        )

        ok = np.abs(den) > 1e-12

        p[ok] = (
            b[ok] - w[ok]
        ) / den[ok]

        vals.append(p)
        done += m

    vals = np.concatenate(vals)
    vals = vals[
        np.isfinite(vals)
    ]

    return tuple(
        np.quantile(
            vals,
            [0.025, 0.975],
        )
    )


print("=" * 100)
print("LAMBDAMART METADATA STRESS")
print("=" * 100)

df = pd.read_csv(CANDIDATES)

df["task_id"] = (
    df["task_id"]
    .astype(str)
)

df["_input_order"] = np.arange(
    len(df)
)

# The frozen candidate matrix stores candidate identity as `doc`
# rather than duplicating lecture_id. Reconstruct candidate lecture
# from the authoritative corpus, then validate the reconstructed
# position features against the already-frozen feature matrix.
CORPUS = (
    ROOT
    / "data/corpus/slide_corpus_final.jsonl"
)

with CORPUS.open(
    "r",
    encoding="utf-8",
) as f:
    corpus_rows = [
        json.loads(line)
        for line in f
        if line.strip()
    ]

doc_to_lecture = {
    str(x["doc_id"]): int(x["lecture_id"])
    for x in corpus_rows
}

if "doc" not in df.columns:
    raise RuntimeError(
        "Frozen candidate matrix lacks required `doc` column."
    )

df["candidate_lecture"] = (
    df["doc"]
    .astype(str)
    .map(doc_to_lecture)
)

if df["candidate_lecture"].isna().any():
    missing = (
        df.loc[
            df["candidate_lecture"].isna(),
            "doc",
        ]
        .astype(str)
        .drop_duplicates()
        .tolist()
    )

    raise RuntimeError(
        "Candidate docs missing from corpus mapping: "
        + repr(missing[:20])
    )

df["candidate_lecture"] = (
    df["candidate_lecture"]
    .astype(int)
)

expected_same = (
    df["candidate_lecture"]
    .eq(
        pd.to_numeric(
            df["target_lecture"]
        ).astype(int)
    )
    .astype(int)
)

expected_distance = (
    df["candidate_lecture"]
    - pd.to_numeric(
        df["target_lecture"]
    ).astype(int)
).abs()

if not np.array_equal(
    expected_same.to_numpy(int),
    df["same_lecture"].to_numpy(int),
):
    raise RuntimeError(
        "Reconstructed candidate lecture does not reproduce "
        "frozen same_lecture."
    )

if not np.allclose(
    expected_distance.to_numpy(float),
    df["lecture_distance"].to_numpy(float),
    atol=1e-12,
    rtol=0,
):
    raise RuntimeError(
        "Reconstructed candidate lecture does not reproduce "
        "frozen lecture_distance."
    )

print(
    "CORPUS DOC->LECTURE RECONSTRUCTION GATE = PASSED"
)

candidate_lecture_col = (
    find_candidate_lecture_column(df)
)

print(
    "Candidate lecture column:",
    candidate_lecture_col,
)

print(
    "Rows:",
    len(df),
)

print(
    "Tasks:",
    df["task_id"].nunique(),
)

print(
    "Lectures:",
    df["target_lecture"].nunique(),
)


# Same outer folds as the nested LambdaMART experiment.
task_ids = np.array(
    sorted(
        df["task_id"].unique()
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

outer = list(
    GroupKFold(
        n_splits=5
    ).split(
        task_ids,
        groups=groups,
    )
)

tuning = pd.read_csv(TUNING)

lecture_universe = set(
    pd.to_numeric(
        df["target_lecture"]
    )
    .astype(int)
    .unique()
    .tolist()
)

correct_parts = []
minus_parts = []
plus_parts = []

for fold, (
    train_ix,
    test_ix,
) in enumerate(
    outer,
    start=1,
):
    train_ids = set(
        task_ids[train_ix]
    )

    test_ids = set(
        task_ids[test_ix]
    )

    train = df[
        df["task_id"].isin(
            train_ids
        )
    ].copy()

    test = df[
        df["task_id"].isin(
            test_ids
        )
    ].copy()

    cand = (
        tuning[
            (
                tuning["outer_fold"]
                == fold
            )
            & (
                tuning["method"]
                == "Full PACER LambdaMART"
            )
        ]
        .sort_values(
            [
                "mean_AnyGold@3",
                "mean_MRRAnyGold",
                "num_leaves",
                "n_estimators",
                "min_child_samples",
            ],
            ascending=[
                False,
                False,
                True,
                True,
                True,
            ],
            kind="mergesort",
        )
    )

    if len(cand) == 0:
        raise RuntimeError(
            f"No tuning rows for fold {fold}"
        )

    best = cand.iloc[0]

    params = {
        "num_leaves":
            int(best["num_leaves"]),
        "min_child_samples":
            int(
                best[
                    "min_child_samples"
                ]
            ),
        "n_estimators":
            int(best["n_estimators"]),
        "learning_rate":
            float(
                best["learning_rate"]
            ),
    }

    print(
        f"Fold {fold}:",
        params,
    )

    model = fit_ranker(
        train,
        FULL_FEATURES,
        params,
    )

    # Correct metadata.
    correct_scored = score(
        model,
        test,
        FULL_FEATURES,
    )

    correct_parts.append(
        per_task(
            correct_scored,
            "Full PACER LambdaMART correct metadata",
        )
    )

    # -1 and +1 test-time feature perturbations.
    for shift, holder, label in [
        (
            -1,
            minus_parts,
            "Full PACER LambdaMART assumed lecture -1",
        ),
        (
            +1,
            plus_parts,
            "Full PACER LambdaMART assumed lecture +1",
        ),
    ]:
        x = test.copy()

        x["assumed_lecture"] = (
            pd.to_numeric(
                x["target_lecture"]
            )
            .astype(int)
            + shift
        )

        valid_tasks = set(
            x.loc[
                x[
                    "assumed_lecture"
                ].isin(
                    lecture_universe
                ),
                "task_id",
            ]
            .unique()
            .tolist()
        )

        x = x[
            x["task_id"].isin(
                valid_tasks
            )
        ].copy()

        cand_lecture = (
            pd.to_numeric(
                x[
                    candidate_lecture_col
                ]
            )
            .astype(int)
        )

        x["same_lecture"] = (
            cand_lecture
            == x["assumed_lecture"]
        ).astype(int)

        x["lecture_distance"] = (
            cand_lecture
            - x["assumed_lecture"]
        ).abs()

        stressed = score(
            model,
            x,
            FULL_FEATURES,
        )

        holder.append(
            per_task(
                stressed,
                label,
            )
        )


correct = pd.concat(
    correct_parts,
    ignore_index=True,
)

minus = pd.concat(
    minus_parts,
    ignore_index=True,
)

plus = pd.concat(
    plus_parts,
    ignore_index=True,
)


# Hard reproduction gate against the already frozen held-out result.
old = pd.read_csv(OLD_METRICS)

old["task_id"] = (
    old["task_id"]
    .astype(str)
)

old_full = (
    old[
        old["method"]
        == "Full PACER LambdaMART"
    ]
    .copy()
)

old_content = (
    old[
        old["method"]
        == "Content-only LambdaMART"
    ]
    .copy()
)

gate = correct.merge(
    old_full[
        [
            "task_id",
            "AnyGold@3",
            "TargetSlide@3",
            "MRRAnyGold",
        ]
    ],
    on="task_id",
    suffixes=(
        "_new",
        "_old",
    ),
    validate="one_to_one",
)

for metric in [
    "AnyGold@3",
    "TargetSlide@3",
    "MRRAnyGold",
]:
    a = gate[
        metric + "_new"
    ].to_numpy(float)

    b = gate[
        metric + "_old"
    ].to_numpy(float)

    if not np.allclose(
        a,
        b,
        atol=1e-12,
        rtol=0,
    ):
        raise RuntimeError(
            f"Correct-metadata reproduction failed for {metric}"
        )

print()
print(
    "CORRECT-METADATA REPRODUCTION GATE = PASSED"
)

print()
print("Correct:")
print(summarize(correct))

print()
print("-1:")
print(summarize(minus))

print()
print("+1:")
print(summarize(plus))


all_conditions = pd.concat(
    [
        old_content.assign(
            condition=(
                "Content-only LambdaMART"
            )
        )[
            [
                "condition",
                "task_id",
                "target_lecture",
                "AnyGold@3",
                "TargetSlide@3",
                "MRRAnyGold",
                "FirstGoldRank",
            ]
        ],
        correct,
        minus,
        plus,
    ],
    ignore_index=True,
)

all_conditions.to_csv(
    OUT
    / "lambdamart_metadata_stress_per_task.csv",
    index=False,
)


# Matched directional comparisons.
comparison_rows = []

for name, wrong in [
    ("Backward -1", minus),
    ("Forward +1", plus),
]:
    matched = (
        correct.merge(
            wrong[
                [
                    "task_id",
                    "AnyGold@3",
                    "TargetSlide@3",
                    "MRRAnyGold",
                ]
            ],
            on="task_id",
            suffixes=(
                "_correct",
                "_wrong",
            ),
            validate="one_to_one",
        )
        .merge(
            old_content[
                [
                    "task_id",
                    "AnyGold@3",
                    "TargetSlide@3",
                    "MRRAnyGold",
                ]
            ],
            on="task_id",
            validate="one_to_one",
        )
    )

    for j, metric in enumerate(
        [
            "AnyGold@3",
            "TargetSlide@3",
            "MRRAnyGold",
        ]
    ):
        ccol = metric + "_correct"
        wcol = metric + "_wrong"
        bcol = metric

        ci_wc = cluster_bootstrap_delta(
            matched,
            wcol,
            ccol,
            N_BOOT,
            SEED + 1000 + j,
        )

        ci_wb = cluster_bootstrap_delta(
            matched,
            wcol,
            bcol,
            N_BOOT,
            SEED + 2000 + j,
        )

        comparison_rows.append(
            {
                "scenario": name,
                "metric": metric,
                "N": len(matched),
                "n_lectures": int(
                    matched[
                        "target_lecture"
                    ].nunique()
                ),
                "correct": float(
                    matched[ccol].mean()
                ),
                "wrong": float(
                    matched[wcol].mean()
                ),
                "content": float(
                    matched[bcol].mean()
                ),
                "wrong_minus_correct":
                    float(
                        (
                            matched[wcol]
                            - matched[ccol]
                        ).mean()
                    ),
                "wrong_minus_correct_ci_low":
                    float(ci_wc[0]),
                "wrong_minus_correct_ci_high":
                    float(ci_wc[1]),
                "wrong_minus_content":
                    float(
                        (
                            matched[wcol]
                            - matched[bcol]
                        ).mean()
                    ),
                "wrong_minus_content_ci_low":
                    float(ci_wb[0]),
                "wrong_minus_content_ci_high":
                    float(ci_wb[1]),
            }
        )


pd.DataFrame(
    comparison_rows
).to_csv(
    OUT
    / "lambdamart_metadata_stress_comparisons.csv",
    index=False,
)


# Reliability operating envelope.
reliability_rows = []


def scenario_frame(
    wrong_df,
):
    z = (
        correct.merge(
            old_content[
                [
                    "task_id",
                    "AnyGold@3",
                    "TargetSlide@3",
                    "MRRAnyGold",
                ]
            ],
            on="task_id",
            suffixes=(
                "_correct",
                "_content",
            ),
            validate="one_to_one",
        )
        .merge(
            wrong_df[
                [
                    "task_id",
                    "AnyGold@3",
                    "TargetSlide@3",
                    "MRRAnyGold",
                ]
            ],
            on="task_id",
            validate="one_to_one",
        )
    )

    return z


for sidx, (
    scenario_name,
    z,
    mode,
) in enumerate(
    [
        (
            "Backward one-lecture error",
            scenario_frame(minus),
            "direct",
        ),
        (
            "Forward one-lecture error",
            scenario_frame(plus),
            "direct",
        ),
    ]
):
    for midx, metric in enumerate(
        [
            "AnyGold@3",
            "TargetSlide@3",
            "MRRAnyGold",
        ]
    ):
        c = metric + "_correct"
        b = metric + "_content"
        w = metric

        cm = float(z[c].mean())
        wm = float(z[w].mean())
        bm = float(z[b].mean())

        p = break_even(
            cm,
            wm,
            bm,
        )

        lo, hi = (
            cluster_bootstrap_break_even(
                z,
                c,
                w,
                b,
                SEED
                + 3000
                + sidx * 100
                + midx,
            )
        )

        reliability_rows.append(
            {
                "scenario":
                    scenario_name,
                "metric":
                    metric,
                "N":
                    len(z),
                "n_lectures":
                    int(
                        z[
                            "target_lecture"
                        ].nunique()
                    ),
                "correct_pacer":
                    cm,
                "wrong_pacer":
                    wm,
                "content_only":
                    bm,
                "break_even_correctness":
                    p,
                "cluster_ci95_low":
                    lo,
                "cluster_ci95_high":
                    hi,
            }
        )


# Symmetric interior subset.
interior_ids = sorted(
    set(minus["task_id"])
    & set(plus["task_id"])
)

sym = (
    correct[
        correct[
            "task_id"
        ].isin(
            interior_ids
        )
    ]
    .merge(
        old_content[
            [
                "task_id",
                "AnyGold@3",
                "TargetSlide@3",
                "MRRAnyGold",
            ]
        ],
        on="task_id",
        suffixes=(
            "_correct",
            "_content",
        ),
        validate="one_to_one",
    )
    .merge(
        minus[
            [
                "task_id",
                "AnyGold@3",
                "TargetSlide@3",
                "MRRAnyGold",
            ]
        ],
        on="task_id",
        suffixes=(
            "",
            "_minus",
        ),
        validate="one_to_one",
    )
    .rename(
        columns={
            "AnyGold@3":
                "AnyGold@3_minus",
            "TargetSlide@3":
                "TargetSlide@3_minus",
            "MRRAnyGold":
                "MRRAnyGold_minus",
        }
    )
    .merge(
        plus[
            [
                "task_id",
                "AnyGold@3",
                "TargetSlide@3",
                "MRRAnyGold",
            ]
        ],
        on="task_id",
        validate="one_to_one",
    )
    .rename(
        columns={
            "AnyGold@3":
                "AnyGold@3_plus",
            "TargetSlide@3":
                "TargetSlide@3_plus",
            "MRRAnyGold":
                "MRRAnyGold_plus",
        }
    )
)

for midx, metric in enumerate(
    [
        "AnyGold@3",
        "TargetSlide@3",
        "MRRAnyGold",
    ]
):
    c = metric + "_correct"
    b = metric + "_content"

    w = metric + "_wrong_mean"

    sym[w] = (
        sym[metric + "_minus"]
        + sym[metric + "_plus"]
    ) / 2.0

    cm = float(sym[c].mean())
    wm = float(sym[w].mean())
    bm = float(sym[b].mean())

    p = break_even(
        cm,
        wm,
        bm,
    )

    lo, hi = (
        cluster_bootstrap_break_even(
            sym,
            c,
            w,
            b,
            SEED + 4000 + midx,
        )
    )

    reliability_rows.append(
        {
            "scenario":
                "Symmetric +/-1 error on interior lectures",
            "metric":
                metric,
            "N":
                len(sym),
            "n_lectures":
                int(
                    sym[
                        "target_lecture"
                    ].nunique()
                ),
            "correct_pacer":
                cm,
            "wrong_pacer":
                wm,
            "content_only":
                bm,
            "break_even_correctness":
                p,
            "cluster_ci95_low":
                lo,
            "cluster_ci95_high":
                hi,
        }
    )


reliability = pd.DataFrame(
    reliability_rows
)

reliability.to_csv(
    OUT
    / "lambdamart_metadata_reliability_break_even.csv",
    index=False,
)

print()
print("=" * 100)
print("LAMBDAMART BREAK-EVEN SUMMARY")
print("=" * 100)
print(
    reliability.to_string(
        index=False
    )
)

metadata = {
    "experiment":
        "Full PACER LambdaMART test-time metadata stress",
    "candidate_pool":
        "unchanged frozen candidate pool",
    "perturbation":
        "test-time same_lecture and lecture_distance features only",
    "candidate_lecture_column":
        candidate_lecture_col,
    "correct_reproduction_gate":
        "PASSED",
    "seed":
        SEED,
    "bootstrap_replicates":
        N_BOOT,
}

(
    OUT
    / "LAMBDAMART_METADATA_STRESS_METADATA.json"
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
    / "LAMBDAMART_METADATA_STRESS_COMPLETE.txt"
).write_text(
    "PASS\n",
    encoding="utf-8",
)

print()
print(
    "LAMBDAMART_METADATA_STRESS_COMPLETE = PASS"
)
