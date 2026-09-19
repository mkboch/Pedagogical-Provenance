#!/usr/bin/env python3

# Public path-portable copy of the frozen analysis source.
# Frozen analysis source SHA256: 873e4829774435fdb8208bae8a9a2fb3a25dfdd0d0c901e35af11adc342e0947

from pathlib import Path
import hashlib
import json
import math

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from scipy.stats import binomtest


ROOT = Path(__file__).resolve().parents[2]
INPUT = ROOT / "artifacts/final_targeted_validation_20260822/position_aware_candidate_features.csv"
OUT = ROOT / "artifacts/jiis_position_ablation_20260918"

SEED = 20260822
N_FOLDS = 5
KS = [1, 3, 5, 10, 20, 50]

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

# Frozen manuscript reproduction gates.
EXPECTED_RRF_ANY3 = 87
EXPECTED_RRF_TARGET3 = 73

EXPECTED_LECTURE_ANY3 = 231
EXPECTED_LECTURE_TARGET3 = 193

EXPECTED_FULL_ANY3 = 314
EXPECTED_FULL_TARGET3 = 269


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def wilson_ci(x, n, z=1.959963984540054):
    if n == 0:
        return (np.nan, np.nan)

    p = x / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2.0 * n)) / denom
    half = (
        z
        * math.sqrt(
            p * (1.0 - p) / n
            + z * z / (4.0 * n * n)
        )
        / denom
    )
    return center - half, center + half


def bootstrap_mean_ci(diff, n_boot=50000, seed=SEED + 17):
    diff = np.asarray(diff, dtype=float)
    rng = np.random.default_rng(seed)

    vals = np.empty(n_boot, dtype=float)
    n = len(diff)
    chunk = 1000

    pos = 0
    while pos < n_boot:
        m = min(chunk, n_boot - pos)
        idx = rng.integers(0, n, size=(m, n))
        vals[pos:pos + m] = diff[idx].mean(axis=1)
        pos += m

    return tuple(np.quantile(vals, [0.025, 0.975]))


def signflip_pvalue(diff, n_resamples=100000, seed=SEED + 23):
    diff = np.asarray(diff, dtype=float)
    observed = abs(diff.mean())

    rng = np.random.default_rng(seed)
    exceed = 0
    done = 0
    chunk = 2000

    while done < n_resamples:
        m = min(chunk, n_resamples - done)
        signs = rng.choice(
            np.array([-1.0, 1.0]),
            size=(m, len(diff)),
        )
        perm = np.abs((signs * diff).mean(axis=1))
        exceed += int(np.sum(perm >= observed - 1e-15))
        done += m

    return (exceed + 1.0) / (n_resamples + 1.0)


def exact_mcnemar(content, full):
    content = np.asarray(content, dtype=int)
    full = np.asarray(full, dtype=int)

    # b: content correct, full wrong
    # c: content wrong, full correct
    b = int(np.sum((content == 1) & (full == 0)))
    c = int(np.sum((content == 0) & (full == 1)))

    if b + c == 0:
        p = 1.0
    else:
        p = float(
            binomtest(
                min(b, c),
                n=b + c,
                p=0.5,
                alternative="two-sided",
            ).pvalue
        )

    return b, c, p


def holm_adjust(pvals):
    names = list(pvals)
    ordered = sorted(names, key=lambda x: pvals[x])

    adjusted = {}
    running = 0.0
    m = len(names)

    for i, name in enumerate(ordered):
        raw_adj = (m - i) * pvals[name]
        running = max(running, raw_adj)
        adjusted[name] = min(1.0, running)

    return adjusted


def make_lr():
    return Pipeline(
        [
            ("scale", StandardScaler()),
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


def score_learned_model(df, task_ids, folds, features, method):
    scored_parts = []
    coef_rows = []

    for fold_index, (train_index, test_index) in enumerate(folds, start=1):
        train_ids = set(task_ids[train_index])
        test_ids = set(task_ids[test_index])

        train = df[df["task_id"].isin(train_ids)]
        test = df[df["task_id"].isin(test_ids)].copy()

        model = make_lr()

        model.fit(
            train[features].to_numpy(float),
            train["is_gold"].to_numpy(int),
        )

        test["score"] = model.predict_proba(
            test[features].to_numpy(float)
        )[:, 1]

        test["method"] = method
        test["fold"] = fold_index
        scored_parts.append(test)

        coefs = model.named_steps["logistic"].coef_[0]

        for feature, coefficient in zip(features, coefs):
            coef_rows.append(
                {
                    "method": method,
                    "fold": fold_index,
                    "feature": feature,
                    "standardized_coefficient": float(coefficient),
                }
            )

    return pd.concat(scored_parts, ignore_index=True), coef_rows


def evaluate_scores(scored, method, score_column="score"):
    task_rows = []
    k_rows = []

    # Preserve the frozen candidate order as the final tie breaker.
    scored = scored.copy()
    scored["_input_order"] = np.arange(len(scored))

    for task_id, group in scored.groupby("task_id", sort=True):

        if method == "RRF baseline":
            ranked = group.sort_values(
                ["rrf_rank", "_input_order"],
                ascending=[True, True],
                kind="mergesort",
            )
        else:
            ranked = group.sort_values(
                [score_column, "_input_order"],
                ascending=[False, True],
                kind="mergesort",
            )

        gold = ranked["is_gold"].to_numpy(int)
        target = ranked["is_target"].to_numpy(int)

        gold_positions = np.flatnonzero(gold == 1)
        target_positions = np.flatnonzero(target == 1)

        first_gold = int(gold_positions[0] + 1) if len(gold_positions) else None
        first_target = int(target_positions[0] + 1) if len(target_positions) else None

        task_rows.append(
            {
                "method": method,
                "task_id": task_id,
                "target_lecture": int(ranked["target_lecture"].iloc[0]),
                "first_gold_rank": first_gold,
                "first_target_rank": first_target,
                "rr_any_gold": 1.0 / first_gold if first_gold else 0.0,
                "rr_target": 1.0 / first_target if first_target else 0.0,
            }
        )

        for k in KS:
            top = ranked.iloc[:k]

            k_rows.append(
                {
                    "method": method,
                    "task_id": task_id,
                    "k": k,
                    "any_gold": int(top["is_gold"].any()),
                    "target_slide": int(top["is_target"].any()),
                }
            )

    return pd.DataFrame(task_rows), pd.DataFrame(k_rows)


print("============================================================")
print("LOAD AND VALIDATE FROZEN CANDIDATE MATRIX")
print("============================================================")

OUT.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(INPUT)
df["task_id"] = df["task_id"].astype(str)

required = {
    "task_id",
    "target_lecture",
    "doc",
    "rrf_rank",
    "rrf_rr",
    "bm25_rr",
    "bm25_score",
    "tfidf_rr",
    "bge_sim",
    "same_lecture",
    "lecture_distance",
    "is_gold",
    "is_target",
}

missing = sorted(required - set(df.columns))
if missing:
    raise RuntimeError(f"Missing required columns: {missing}")

if len(df) != EXPECTED_ROWS:
    raise RuntimeError(
        f"Frozen row-count mismatch: expected {EXPECTED_ROWS}, found {len(df)}"
    )

n_tasks = df["task_id"].nunique()
if n_tasks != EXPECTED_TASKS:
    raise RuntimeError(
        f"Frozen task-count mismatch: expected {EXPECTED_TASKS}, found {n_tasks}"
    )

if set(FULL_FEATURES) & set(FORBIDDEN_FEATURES):
    raise RuntimeError("Leakage-safe feature set contains forbidden position feature.")

print("input:", INPUT)
print("sha256:", sha256(INPUT))
print("rows:", len(df))
print("tasks:", n_tasks)
print("content features:", CONTENT_FEATURES)
print("position features:", POSITION_FEATURES)
print("full features:", FULL_FEATURES)
print("forbidden exact-position features:", FORBIDDEN_FEATURES)

print()
print("============================================================")
print("RECONSTRUCT ORIGINAL 5-FOLD TARGET-LECTURE SPLIT")
print("============================================================")

task_ids = np.array(sorted(df["task_id"].unique()))

task_lecture = (
    df.groupby("task_id", sort=True)["target_lecture"]
    .first()
)

group_values = np.array(
    [task_lecture.loc[task_id] for task_id in task_ids]
)

cv = GroupKFold(n_splits=N_FOLDS)

folds = list(
    cv.split(
        task_ids,
        groups=group_values,
    )
)

fold_rows = []

for fold_index, (train, test) in enumerate(folds, start=1):
    train_lectures = set(group_values[train])
    test_lectures = set(group_values[test])

    overlap = train_lectures & test_lectures

    fold_rows.append(
        {
            "fold": fold_index,
            "n_train_tasks": len(train),
            "n_test_tasks": len(test),
            "n_train_lectures": len(train_lectures),
            "n_test_lectures": len(test_lectures),
            "lecture_overlap_count": len(overlap),
            "leakage_check_disjoint_lectures": len(overlap) == 0,
        }
    )

    print(
        f"fold {fold_index}: "
        f"train_tasks={len(train)} "
        f"test_tasks={len(test)} "
        f"train_lectures={len(train_lectures)} "
        f"test_lectures={len(test_lectures)} "
        f"overlap={len(overlap)}"
    )

fold_df = pd.DataFrame(fold_rows)
fold_df.to_csv(OUT / "fold_audit.csv", index=False)

if not fold_df["leakage_check_disjoint_lectures"].all():
    raise RuntimeError("Target-lecture split leakage detected.")

print()
print("============================================================")
print("FIT MATCHED ABLATION MODELS")
print("============================================================")

all_task_metrics = []
all_k_metrics = []
all_coefficients = []

# ------------------------------------------------------------
# 1. Plain RRF baseline
# ------------------------------------------------------------
rrf = df.copy()
rrf["score"] = -rrf["rrf_rank"].astype(float)

task_m, k_m = evaluate_scores(
    rrf,
    "RRF baseline",
)

all_task_metrics.append(task_m)
all_k_metrics.append(k_m)

# ------------------------------------------------------------
# 2. Frozen coarse lecture-first score, w=1.60
# ------------------------------------------------------------
lecture_first = df.copy()

lecture_first["score"] = (
    lecture_first["rrf_rr"].astype(float)
    + 1.60
    * (
        lecture_first["same_lecture"].astype(float)
        - 0.01 * lecture_first["lecture_distance"].astype(float)
    )
)

task_m, k_m = evaluate_scores(
    lecture_first,
    "Coarse lecture-first (w=1.60)",
)

all_task_metrics.append(task_m)
all_k_metrics.append(k_m)

# ------------------------------------------------------------
# 3. Content-only logistic ranker
# ------------------------------------------------------------
content_scored, coefs = score_learned_model(
    df,
    task_ids,
    folds,
    CONTENT_FEATURES,
    "Content-only LR",
)

task_m, k_m = evaluate_scores(
    content_scored,
    "Content-only LR",
)

all_task_metrics.append(task_m)
all_k_metrics.append(k_m)
all_coefficients.extend(coefs)

# ------------------------------------------------------------
# 4. Position-only logistic ranker
# ------------------------------------------------------------
position_scored, coefs = score_learned_model(
    df,
    task_ids,
    folds,
    POSITION_FEATURES,
    "Position-only LR",
)

task_m, k_m = evaluate_scores(
    position_scored,
    "Position-only LR",
)

all_task_metrics.append(task_m)
all_k_metrics.append(k_m)
all_coefficients.extend(coefs)

# ------------------------------------------------------------
# 5. Full PACER leakage-safe logistic ranker
# ------------------------------------------------------------
full_scored, coefs = score_learned_model(
    df,
    task_ids,
    folds,
    FULL_FEATURES,
    "PACER full LR",
)

task_m, k_m = evaluate_scores(
    full_scored,
    "PACER full LR",
)

all_task_metrics.append(task_m)
all_k_metrics.append(k_m)
all_coefficients.extend(coefs)

task_metrics = pd.concat(all_task_metrics, ignore_index=True)
k_metrics = pd.concat(all_k_metrics, ignore_index=True)

pd.DataFrame(all_coefficients).to_csv(
    OUT / "fold_standardized_coefficients.csv",
    index=False,
)

task_metrics.to_csv(
    OUT / "per_task_rank_metrics.csv",
    index=False,
)

k_metrics.to_csv(
    OUT / "per_task_k_metrics.csv",
    index=False,
)

print("models completed")

print()
print("============================================================")
print("AGGREGATE RESULTS")
print("============================================================")

k_summary = (
    k_metrics
    .groupby(["method", "k"], as_index=False)
    .agg(
        N=("task_id", "nunique"),
        AnyGold_hits=("any_gold", "sum"),
        AnyGold_at_k=("any_gold", "mean"),
        TargetSlide_hits=("target_slide", "sum"),
        TargetSlide_at_k=("target_slide", "mean"),
    )
)

mrr_summary = (
    task_metrics
    .groupby("method", as_index=False)
    .agg(
        N=("task_id", "nunique"),
        MRRAnyGold=("rr_any_gold", "mean"),
        MRRTargetSlide=("rr_target", "mean"),
        MedianFirstGoldRank=("first_gold_rank", "median"),
        MedianFirstTargetRank=("first_target_rank", "median"),
    )
)

summary3 = k_summary[k_summary["k"] == 3].merge(
    mrr_summary,
    on=["method", "N"],
    how="left",
)

wilson_rows = []

for _, row in summary3.iterrows():
    lo_a, hi_a = wilson_ci(
        int(row["AnyGold_hits"]),
        int(row["N"]),
    )
    lo_t, hi_t = wilson_ci(
        int(row["TargetSlide_hits"]),
        int(row["N"]),
    )

    wilson_rows.append(
        {
            **row.to_dict(),
            "AnyGold_at_3_CI_low": lo_a,
            "AnyGold_at_3_CI_high": hi_a,
            "TargetSlide_at_3_CI_low": lo_t,
            "TargetSlide_at_3_CI_high": hi_t,
        }
    )

summary3 = pd.DataFrame(wilson_rows)

method_order = [
    "RRF baseline",
    "Coarse lecture-first (w=1.60)",
    "Content-only LR",
    "Position-only LR",
    "PACER full LR",
]

summary3["method"] = pd.Categorical(
    summary3["method"],
    categories=method_order,
    ordered=True,
)

summary3 = summary3.sort_values("method").reset_index(drop=True)

summary3.to_csv(
    OUT / "ablation_summary_at3.csv",
    index=False,
)

k_summary.to_csv(
    OUT / "ablation_k_sweep.csv",
    index=False,
)

mrr_summary.to_csv(
    OUT / "ablation_mrr.csv",
    index=False,
)

print(
    summary3[
        [
            "method",
            "AnyGold_hits",
            "AnyGold_at_k",
            "TargetSlide_hits",
            "TargetSlide_at_k",
            "MRRAnyGold",
            "MRRTargetSlide",
            "MedianFirstGoldRank",
        ]
    ].to_string(index=False)
)

print()
print("============================================================")
print("FROZEN REPRODUCTION GATES")
print("============================================================")

def get_at3(method):
    row = summary3[summary3["method"].astype(str) == method].iloc[0]
    return int(row["AnyGold_hits"]), int(row["TargetSlide_hits"])


rrf_any, rrf_target = get_at3("RRF baseline")
lec_any, lec_target = get_at3("Coarse lecture-first (w=1.60)")
full_any, full_target = get_at3("PACER full LR")

checks = [
    ("RRF AnyGold@3", rrf_any, EXPECTED_RRF_ANY3),
    ("RRF TargetSlide@3", rrf_target, EXPECTED_RRF_TARGET3),
    ("Lecture-first AnyGold@3", lec_any, EXPECTED_LECTURE_ANY3),
    ("Lecture-first TargetSlide@3", lec_target, EXPECTED_LECTURE_TARGET3),
    ("Full PACER AnyGold@3", full_any, EXPECTED_FULL_ANY3),
    ("Full PACER TargetSlide@3", full_target, EXPECTED_FULL_TARGET3),
]

all_pass = True

for name, observed, expected in checks:
    ok = observed == expected
    all_pass &= ok
    print(
        f"{name}: observed={observed} expected={expected} "
        f"{'PASS' if ok else 'FAIL'}"
    )

if not all_pass:
    raise RuntimeError(
        "Frozen reproduction gate failed. "
        "Do not interpret new ablations until this is resolved."
    )

print("FROZEN REPRODUCTION: PASS")

print()
print("============================================================")
print("PAIRED FULL PACER VS CONTENT-ONLY TESTS")
print("============================================================")

k3 = k_metrics[k_metrics["k"] == 3]

any_pivot = k3.pivot(
    index="task_id",
    columns="method",
    values="any_gold",
)

target_pivot = k3.pivot(
    index="task_id",
    columns="method",
    values="target_slide",
)

mrr_pivot = task_metrics.pivot(
    index="task_id",
    columns="method",
    values="rr_any_gold",
)

content_any = any_pivot["Content-only LR"].to_numpy(int)
full_any_arr = any_pivot["PACER full LR"].to_numpy(int)

content_target = target_pivot["Content-only LR"].to_numpy(int)
full_target_arr = target_pivot["PACER full LR"].to_numpy(int)

content_mrr = mrr_pivot["Content-only LR"].to_numpy(float)
full_mrr = mrr_pivot["PACER full LR"].to_numpy(float)

# Binary @3 outcomes
b_any, c_any, p_any = exact_mcnemar(
    content_any,
    full_any_arr,
)

diff_any = full_any_arr - content_any
ci_any = bootstrap_mean_ci(diff_any)

b_target, c_target, p_target = exact_mcnemar(
    content_target,
    full_target_arr,
)

diff_target = full_target_arr - content_target
ci_target = bootstrap_mean_ci(
    diff_target,
    seed=SEED + 18,
)

# Continuous paired MRR
diff_mrr = full_mrr - content_mrr
ci_mrr = bootstrap_mean_ci(
    diff_mrr,
    seed=SEED + 19,
)
p_mrr = signflip_pvalue(diff_mrr)

raw_p = {
    "AnyGold@3": p_any,
    "TargetSlide@3": p_target,
    "MRRAnyGold": p_mrr,
}

holm = holm_adjust(raw_p)

paired_rows = [
    {
        "comparison": "PACER full LR - Content-only LR",
        "metric": "AnyGold@3",
        "content_only": float(content_any.mean()),
        "full_pacer": float(full_any_arr.mean()),
        "delta": float(diff_any.mean()),
        "ci95_low": float(ci_any[0]),
        "ci95_high": float(ci_any[1]),
        "discordant_content_better": b_any,
        "discordant_pacer_better": c_any,
        "test": "exact McNemar",
        "p_raw": p_any,
        "p_holm_3_tests": holm["AnyGold@3"],
    },
    {
        "comparison": "PACER full LR - Content-only LR",
        "metric": "TargetSlide@3",
        "content_only": float(content_target.mean()),
        "full_pacer": float(full_target_arr.mean()),
        "delta": float(diff_target.mean()),
        "ci95_low": float(ci_target[0]),
        "ci95_high": float(ci_target[1]),
        "discordant_content_better": b_target,
        "discordant_pacer_better": c_target,
        "test": "exact McNemar",
        "p_raw": p_target,
        "p_holm_3_tests": holm["TargetSlide@3"],
    },
    {
        "comparison": "PACER full LR - Content-only LR",
        "metric": "MRRAnyGold",
        "content_only": float(content_mrr.mean()),
        "full_pacer": float(full_mrr.mean()),
        "delta": float(diff_mrr.mean()),
        "ci95_low": float(ci_mrr[0]),
        "ci95_high": float(ci_mrr[1]),
        "discordant_content_better": np.nan,
        "discordant_pacer_better": np.nan,
        "test": "paired sign-flip permutation",
        "p_raw": p_mrr,
        "p_holm_3_tests": holm["MRRAnyGold"],
    },
]

paired = pd.DataFrame(paired_rows)

paired.to_csv(
    OUT / "full_vs_content_paired_tests.csv",
    index=False,
)

print(paired.to_string(index=False))

print()
print("============================================================")
print("SAVE MANUSCRIPT-READY REPORT")
print("============================================================")

coef_df = pd.DataFrame(all_coefficients)

coef_summary = (
    coef_df
    .groupby(["method", "feature"], as_index=False)
    .agg(
        mean_standardized_coefficient=("standardized_coefficient", "mean"),
        sd_standardized_coefficient=("standardized_coefficient", "std"),
    )
)

coef_summary.to_csv(
    OUT / "coefficient_summary.csv",
    index=False,
)

report_lines = []

report_lines.append("# PACER JIIS Position-Feature Ablation")
report_lines.append("")
report_lines.append(f"- Input: `{INPUT}`")
report_lines.append(f"- Input SHA-256: `{sha256(INPUT)}`")
report_lines.append(f"- Candidate rows: {len(df)}")
report_lines.append(f"- Tasks: {n_tasks}")
report_lines.append(f"- CV: {N_FOLDS}-fold GroupKFold by target lecture")
report_lines.append(f"- Seed: {SEED}")
report_lines.append("")
report_lines.append("## Features")
report_lines.append("")
report_lines.append(f"- Content-only: `{', '.join(CONTENT_FEATURES)}`")
report_lines.append(f"- Position-only: `{', '.join(POSITION_FEATURES)}`")
report_lines.append(f"- Full PACER: `{', '.join(FULL_FEATURES)}`")
report_lines.append(
    f"- Explicitly excluded exact-position features: "
    f"`{', '.join(FORBIDDEN_FEATURES)}`"
)
report_lines.append("")
report_lines.append("## Primary results at k=3")
report_lines.append("")
report_lines.append(
    summary3[
        [
            "method",
            "AnyGold_hits",
            "AnyGold_at_k",
            "TargetSlide_hits",
            "TargetSlide_at_k",
            "MRRAnyGold",
            "MedianFirstGoldRank",
        ]
    ].to_markdown(index=False)
)
report_lines.append("")
report_lines.append("## PACER full versus content-only")
report_lines.append("")
report_lines.append(paired.to_markdown(index=False))
report_lines.append("")
report_lines.append("## Reproduction gate")
report_lines.append("")
report_lines.append("- RRF frozen headline: PASS")
report_lines.append("- Coarse lecture-first frozen headline: PASS")
report_lines.append("- Full leakage-safe PACER frozen headline: PASS")
report_lines.append("")
report_lines.append(
    "Interpretation must remain conditional on accurate current-lecture "
    "identification because target lecture equals current lecture by benchmark "
    "construction."
)

(OUT / "RESULTS.md").write_text(
    "\n".join(report_lines) + "\n",
    encoding="utf-8",
)

metadata = {
    "input": str(INPUT),
    "input_sha256": sha256(INPUT),
    "rows": len(df),
    "tasks": int(n_tasks),
    "seed": SEED,
    "n_folds": N_FOLDS,
    "content_features": CONTENT_FEATURES,
    "position_features": POSITION_FEATURES,
    "full_features": FULL_FEATURES,
    "forbidden_exact_position_features": FORBIDDEN_FEATURES,
    "frozen_reproduction_pass": True,
}

(OUT / "RUN_METADATA.json").write_text(
    json.dumps(metadata, indent=2) + "\n",
    encoding="utf-8",
)

print("wrote:", OUT / "RESULTS.md")
print("wrote:", OUT / "ablation_summary_at3.csv")
print("wrote:", OUT / "ablation_k_sweep.csv")
print("wrote:", OUT / "ablation_mrr.csv")
print("wrote:", OUT / "full_vs_content_paired_tests.csv")
print("wrote:", OUT / "fold_audit.csv")
print("wrote:", OUT / "coefficient_summary.csv")
print("wrote:", OUT / "per_task_rank_metrics.csv")

print()
print("============================================================")
print("EXPERIMENT COMPLETE")
print("============================================================")
