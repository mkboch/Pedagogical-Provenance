#!/usr/bin/env python3

# Public path-portable copy of the frozen analysis source.
# Frozen analysis source SHA256: 23d85523f68eb4f775017dfd778bbadae0b775e38a6b5ea7a3adf5670ec513c3

from pathlib import Path
import json

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "artifacts/jiis_position_ablation_20260918"

SEED = 20260822
N_BOOT = 100000
N_PERM = 1000000

TASK_METRICS = OUT / "per_task_rank_metrics.csv"
K_METRICS = OUT / "per_task_k_metrics.csv"



OUT.mkdir(parents=True, exist_ok=True)

def holm_adjust(pvals):
    names = list(pvals)
    ordered = sorted(names, key=lambda x: pvals[x])

    adjusted = {}
    running = 0.0
    m = len(names)

    for i, name in enumerate(ordered):
        value = (m - i) * pvals[name]
        running = max(running, value)
        adjusted[name] = min(1.0, running)

    return adjusted


def cluster_bootstrap_ci(cluster_sum, cluster_n, n_boot, seed):
    cluster_sum = np.asarray(cluster_sum, dtype=float)
    cluster_n = np.asarray(cluster_n, dtype=float)

    rng = np.random.default_rng(seed)
    c = len(cluster_sum)

    values = np.empty(n_boot, dtype=float)

    chunk = 5000
    done = 0

    while done < n_boot:
        m = min(chunk, n_boot - done)

        idx = rng.integers(
            0,
            c,
            size=(m, c),
        )

        numer = cluster_sum[idx].sum(axis=1)
        denom = cluster_n[idx].sum(axis=1)

        values[done:done + m] = numer / denom
        done += m

    return tuple(
        np.quantile(
            values,
            [0.025, 0.975],
        )
    )


def cluster_signflip_pvalue(cluster_sum, total_n, observed, n_perm, seed):
    cluster_sum = np.asarray(cluster_sum, dtype=float)

    rng = np.random.default_rng(seed)

    exceed = 0
    done = 0
    c = len(cluster_sum)

    chunk = 10000

    while done < n_perm:
        m = min(chunk, n_perm - done)

        signs = rng.choice(
            np.array([-1.0, 1.0]),
            size=(m, c),
        )

        perm_delta = (
            signs * cluster_sum
        ).sum(axis=1) / total_n

        exceed += int(
            np.sum(
                np.abs(perm_delta)
                >= abs(observed) - 1e-15
            )
        )

        done += m

    return (exceed + 1.0) / (n_perm + 1.0)


task = pd.read_csv(TASK_METRICS)
k = pd.read_csv(K_METRICS)

lecture_map = (
    task[
        ["task_id", "target_lecture"]
    ]
    .drop_duplicates()
)

if lecture_map["task_id"].duplicated().any():
    raise RuntimeError(
        "A task maps to more than one target lecture."
    )

k = k.merge(
    lecture_map,
    on="task_id",
    how="left",
    validate="many_to_one",
)

k3 = k[k["k"] == 3].copy()

print("============================================================")
print("CLUSTER-AWARE PACER INFERENCE")
print("============================================================")
print("tasks:", lecture_map["task_id"].nunique())
print("lectures:", lecture_map["target_lecture"].nunique())
print("cluster unit: target lecture")
print("bootstrap replicates:", N_BOOT)
print("sign-flip permutations:", N_PERM)

comparisons = [
    ("Content-only LR", "PACER full LR"),
    ("Position-only LR", "PACER full LR"),
]

metrics = [
    ("AnyGold@3", k3, "any_gold"),
    ("TargetSlide@3", k3, "target_slide"),
    ("MRRAnyGold", task, "rr_any_gold"),
]

results = []
lecture_detail = []

raw_pvals = {}

for comp_index, (baseline, pacer) in enumerate(comparisons):

    print()
    print("============================================================")
    print(f"{pacer} VS {baseline}")
    print("============================================================")

    for metric_index, (metric_name, frame, value_col) in enumerate(metrics):

        pivot = frame.pivot(
            index="task_id",
            columns="method",
            values=value_col,
        )

        needed = [baseline, pacer]

        if not all(x in pivot.columns for x in needed):
            raise RuntimeError(
                f"Missing method for {metric_name}: {needed}"
            )

        tmp = (
            pivot[needed]
            .reset_index()
            .merge(
                lecture_map,
                on="task_id",
                how="left",
                validate="one_to_one",
            )
        )

        tmp["diff"] = (
            tmp[pacer].astype(float)
            - tmp[baseline].astype(float)
        )

        observed = float(tmp["diff"].mean())

        clustered = (
            tmp.groupby(
                "target_lecture",
                as_index=False,
            )
            .agg(
                n_tasks=("task_id", "size"),
                baseline_mean=(baseline, "mean"),
                pacer_mean=(pacer, "mean"),
                diff_mean=("diff", "mean"),
                diff_sum=("diff", "sum"),
            )
            .sort_values("target_lecture")
        )

        for _, row in clustered.iterrows():
            lecture_detail.append(
                {
                    "baseline": baseline,
                    "pacer": pacer,
                    "metric": metric_name,
                    **row.to_dict(),
                }
            )

        cluster_sum = clustered["diff_sum"].to_numpy(float)
        cluster_n = clustered["n_tasks"].to_numpy(float)

        ci_low, ci_high = cluster_bootstrap_ci(
            cluster_sum,
            cluster_n,
            N_BOOT,
            SEED
            + 100 * comp_index
            + 10 * metric_index
            + 1,
        )

        p = cluster_signflip_pvalue(
            cluster_sum,
            int(cluster_n.sum()),
            observed,
            N_PERM,
            SEED
            + 100 * comp_index
            + 10 * metric_index
            + 2,
        )

        improved = int(
            (clustered["diff_mean"] > 0).sum()
        )
        tied = int(
            np.isclose(
                clustered["diff_mean"],
                0.0,
                atol=1e-15,
            ).sum()
        )
        worse = int(
            (clustered["diff_mean"] < 0).sum()
        )

        unweighted_lecture_delta = float(
            clustered["diff_mean"].mean()
        )

        key = f"{baseline}::{metric_name}"
        raw_pvals[key] = p

        result = {
            "baseline": baseline,
            "pacer": pacer,
            "metric": metric_name,
            "n_tasks": len(tmp),
            "n_lectures": len(clustered),
            "baseline_task_mean": float(
                tmp[baseline].mean()
            ),
            "pacer_task_mean": float(
                tmp[pacer].mean()
            ),
            "task_weighted_delta": observed,
            "cluster_bootstrap_ci95_low": float(ci_low),
            "cluster_bootstrap_ci95_high": float(ci_high),
            "unweighted_mean_lecture_delta":
                unweighted_lecture_delta,
            "lectures_improved": improved,
            "lectures_tied": tied,
            "lectures_worse": worse,
            "cluster_signflip_p_raw": p,
        }

        results.append(result)

        print()
        print(metric_name)
        print(
            f"  baseline = "
            f"{result['baseline_task_mean']:.6f}"
        )
        print(
            f"  PACER    = "
            f"{result['pacer_task_mean']:.6f}"
        )
        print(
            f"  delta    = "
            f"{observed:+.6f}"
        )
        print(
            "  lecture-cluster bootstrap 95% CI = "
            f"[{ci_low:+.6f}, {ci_high:+.6f}]"
        )
        print(
            "  lectures improved/tied/worse = "
            f"{improved}/{tied}/{worse}"
        )
        print(
            "  cluster sign-flip p = "
            f"{p:.8g}"
        )


adjusted = holm_adjust(raw_pvals)

for row in results:
    key = f"{row['baseline']}::{row['metric']}"
    row["cluster_signflip_p_holm_6_tests"] = adjusted[key]


results_df = pd.DataFrame(results)

results_df.to_csv(
    OUT / "cluster_aware_inference.csv",
    index=False,
)

lecture_df = pd.DataFrame(lecture_detail)

lecture_df.to_csv(
    OUT / "per_lecture_deltas.csv",
    index=False,
)


print()
print("============================================================")
print("HOLM-ADJUSTED CLUSTER-AWARE RESULTS")
print("============================================================")

print(
    results_df[
        [
            "baseline",
            "metric",
            "baseline_task_mean",
            "pacer_task_mean",
            "task_weighted_delta",
            "cluster_bootstrap_ci95_low",
            "cluster_bootstrap_ci95_high",
            "lectures_improved",
            "lectures_tied",
            "lectures_worse",
            "cluster_signflip_p_raw",
            "cluster_signflip_p_holm_6_tests",
        ]
    ].to_string(index=False)
)


print()
print("============================================================")
print("FEATURE COEFFICIENT SUMMARY")
print("============================================================")

coef_path = OUT / "coefficient_summary.csv"

if coef_path.exists():
    coef = pd.read_csv(coef_path)

    for method in [
        "Content-only LR",
        "Position-only LR",
        "PACER full LR",
    ]:
        print()
        print(method)

        sub = coef[
            coef["method"] == method
        ].sort_values(
            "mean_standardized_coefficient",
            ascending=False,
        )

        print(
            sub.to_string(
                index=False
            )
        )


report = {
    "cluster_unit": "target_lecture",
    "n_lectures": int(
        lecture_map["target_lecture"].nunique()
    ),
    "n_tasks": int(
        lecture_map["task_id"].nunique()
    ),
    "bootstrap_replicates": N_BOOT,
    "signflip_permutations": N_PERM,
    "holm_family_size": len(raw_pvals),
}

(
    OUT / "CLUSTER_INFERENCE_METADATA.json"
).write_text(
    json.dumps(report, indent=2) + "\n",
    encoding="utf-8",
)

print()
print("wrote:", OUT / "cluster_aware_inference.csv")
print("wrote:", OUT / "per_lecture_deltas.csv")
print(
    "wrote:",
    OUT / "CLUSTER_INFERENCE_METADATA.json",
)

print()
print("============================================================")
print("CLUSTER-AWARE ANALYSIS COMPLETE")
print("============================================================")
