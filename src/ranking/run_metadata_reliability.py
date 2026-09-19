#!/usr/bin/env python3

# Public path-portable copy of the frozen analysis source.
# Frozen analysis source SHA256: 150d8a5863fceedba6844583b4caed403fb694b9762b96a32a6133e12755c327

from pathlib import Path
import json

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

OLD = (
    ROOT
    / "artifacts/jiis_position_ablation_20260918"
)

OUT = (
    ROOT
    / "artifacts/jiis_lambdamart_reliability_20260918"
)

INPUT = (
    OLD
    / "metadata_stress_per_task_metrics.csv"
)

SEED = 20260822
N_BOOT = 100000


METRICS = {
    "AnyGold@3":
        "any_gold_at_3",
    "TargetSlide@3":
        "target_slide_at_3",
    "MRRAnyGold":
        "mrr_any_gold",
}



OUT.mkdir(parents=True, exist_ok=True)

def cluster_bootstrap_break_even(
    frame,
    correct_col,
    wrong_col,
    content_col,
    n_boot,
    seed,
):
    # Aggregate task-level sums within
    # true target-lecture clusters.
    by_lecture = (
        frame.groupby(
            "target_lecture",
            as_index=False,
        )
        .agg(
            n_tasks=("task_id", "size"),
            correct_sum=(
                correct_col,
                "sum",
            ),
            wrong_sum=(
                wrong_col,
                "sum",
            ),
            content_sum=(
                content_col,
                "sum",
            ),
        )
        .sort_values(
            "target_lecture"
        )
    )

    arr = by_lecture[
        [
            "n_tasks",
            "correct_sum",
            "wrong_sum",
            "content_sum",
        ]
    ].to_numpy(float)

    rng = np.random.default_rng(
        seed
    )

    c = len(
        by_lecture
    )

    values = []

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

        sampled = arr[
            idx
        ].sum(
            axis=1
        )

        n = sampled[:, 0]

        correct = (
            sampled[:, 1]
            / n
        )

        wrong = (
            sampled[:, 2]
            / n
        )

        content = (
            sampled[:, 3]
            / n
        )

        denom = (
            correct
            - wrong
        )

        ok = (
            np.abs(denom)
            > 1e-12
        )

        p = np.full(
            m,
            np.nan,
            dtype=float,
        )

        p[ok] = (
            content[ok]
            - wrong[ok]
        ) / denom[ok]

        values.append(
            p
        )

        done += m

    values = np.concatenate(
        values
    )

    values = values[
        np.isfinite(
            values
        )
    ]

    return (
        float(
            np.quantile(
                values,
                0.025,
            )
        ),
        float(
            np.quantile(
                values,
                0.975,
            )
        ),
    )


def break_even(
    correct,
    wrong,
    content,
):
    denom = (
        correct
        - wrong
    )

    if abs(denom) < 1e-12:
        return np.nan

    return (
        content
        - wrong
    ) / denom


df = pd.read_csv(
    INPUT
)

df["task_id"] = (
    df["task_id"]
    .astype(str)
)

print(
    "============================================================"
)
print(
    "PACER METADATA-RELIABILITY OPERATING ENVELOPE"
)
print(
    "============================================================"
)

print(
    "input:",
    INPUT,
)

print(
    "rows:",
    len(df),
)

print(
    "conditions:"
)

for x in sorted(
    df[
        "condition"
    ].unique()
):
    print(
        "  ",
        x,
    )


content = (
    df[
        df["condition"]
        == "Content-only LR"
    ]
    .copy()
)

correct = (
    df[
        df["condition"]
        == "PACER correct metadata"
    ]
    .copy()
)

minus = (
    df[
        df["condition"]
        == "PACER assumed lecture -1"
    ]
    .copy()
)

plus = (
    df[
        df["condition"]
        == "PACER assumed lecture +1"
    ]
    .copy()
)


def merged_direction(
    wrong_frame,
):
    return (
        correct.merge(
            content[
                [
                    "task_id",
                    *METRICS.values(),
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
            wrong_frame[
                [
                    "task_id",
                    *METRICS.values(),
                ]
            ],
            on="task_id",
            how="inner",
            validate="one_to_one",
        )
    )


minus_m = merged_direction(
    minus
)

plus_m = merged_direction(
    plus
)


# Interior tasks where both -1 and +1
# are valid.
interior_ids = sorted(
    set(
        minus[
            "task_id"
        ]
    )
    & set(
        plus[
            "task_id"
        ]
    )
)

interior = (
    correct[
        correct["task_id"]
        .isin(
            interior_ids
        )
    ]
    .copy()
)

interior = interior.merge(
    content[
        [
            "task_id",
            *METRICS.values(),
        ]
    ],
    on="task_id",
    suffixes=(
        "_correct",
        "_content",
    ),
    validate="one_to_one",
)

interior = interior.merge(
    minus[
        [
            "task_id",
            *METRICS.values(),
        ]
    ],
    on="task_id",
    how="inner",
    validate="one_to_one",
)

minus_names = {}

for col in METRICS.values():
    old_name = col
    new_name = (
        col
        + "_minus"
    )

    minus_names[
        old_name
    ] = new_name

interior = interior.rename(
    columns=minus_names
)

interior = interior.merge(
    plus[
        [
            "task_id",
            *METRICS.values(),
        ]
    ],
    on="task_id",
    how="inner",
    validate="one_to_one",
)

plus_names = {}

for col in METRICS.values():
    plus_names[
        col
    ] = (
        col
        + "_plus"
    )

interior = interior.rename(
    columns=plus_names
)


summary_rows = []
curve_rows = []


def analyze_scenario(
    name,
    frame,
    wrong_mode,
    metric_index_offset,
):

    print()
    print(
        "============================================================"
    )
    print(
        name
    )
    print(
        "============================================================"
    )

    for metric_index, (
        metric_name,
        base_col,
    ) in enumerate(
        METRICS.items()
    ):

        correct_col = (
            base_col
            + "_correct"
        )

        content_col = (
            base_col
            + "_content"
        )

        tmp = frame.copy()

        if wrong_mode == "direct":
            wrong_col = base_col

        elif wrong_mode == "symmetric":

            minus_col = (
                base_col
                + "_minus"
            )

            plus_col = (
                base_col
                + "_plus"
            )

            wrong_col = (
                base_col
                + "_wrong_mean"
            )

            tmp[
                wrong_col
            ] = (
                tmp[
                    minus_col
                ]
                + tmp[
                    plus_col
                ]
            ) / 2.0

        else:
            raise ValueError(
                wrong_mode
            )

        correct_mean = float(
            tmp[
                correct_col
            ].mean()
        )

        wrong_mean = float(
            tmp[
                wrong_col
            ].mean()
        )

        content_mean = float(
            tmp[
                content_col
            ].mean()
        )

        p_star = break_even(
            correct_mean,
            wrong_mean,
            content_mean,
        )

        (
            ci_low,
            ci_high,
        ) = cluster_bootstrap_break_even(
            tmp,
            correct_col,
            wrong_col,
            content_col,
            N_BOOT,
            (
                SEED
                + metric_index_offset
                + metric_index
            ),
        )

        print()
        print(
            metric_name
        )
        print(
            f"  N = {len(tmp)}"
        )
        print(
            f"  lectures = "
            f"{tmp['target_lecture'].nunique()}"
        )
        print(
            f"  correct PACER = "
            f"{correct_mean:.6f}"
        )
        print(
            f"  wrong-position PACER = "
            f"{wrong_mean:.6f}"
        )
        print(
            f"  content-only = "
            f"{content_mean:.6f}"
        )
        print(
            f"  break-even metadata correctness = "
            f"{p_star:.6f}"
        )
        print(
            "  cluster-bootstrap 95% CI = "
            f"[{ci_low:.6f}, "
            f"{ci_high:.6f}]"
        )

        summary_rows.append(
            {
                "scenario": name,
                "metric": metric_name,
                "N": len(tmp),
                "n_lectures":
                    tmp[
                        "target_lecture"
                    ].nunique(),
                "correct_pacer":
                    correct_mean,
                "wrong_pacer":
                    wrong_mean,
                "content_only":
                    content_mean,
                "break_even_correctness":
                    p_star,
                "cluster_ci95_low":
                    ci_low,
                "cluster_ci95_high":
                    ci_high,
            }
        )

        for p in np.linspace(
            0.0,
            1.0,
            101,
        ):
            expected = (
                p
                * correct_mean
                + (
                    1.0 - p
                )
                * wrong_mean
            )

            curve_rows.append(
                {
                    "scenario":
                        name,
                    "metric":
                        metric_name,
                    "metadata_correctness":
                        float(p),
                    "expected_pacer":
                        float(expected),
                    "content_only":
                        content_mean,
                    "delta_vs_content":
                        float(
                            expected
                            - content_mean
                        ),
                }
            )


analyze_scenario(
    "Backward one-lecture error",
    minus_m,
    "direct",
    100,
)

analyze_scenario(
    "Forward one-lecture error",
    plus_m,
    "direct",
    200,
)

analyze_scenario(
    "Symmetric +/-1 error on interior lectures",
    interior,
    "symmetric",
    300,
)


summary = pd.DataFrame(
    summary_rows
)

curve = pd.DataFrame(
    curve_rows
)

summary.to_csv(
    OUT
    / "metadata_reliability_break_even.csv",
    index=False,
)

curve.to_csv(
    OUT
    / "metadata_reliability_operating_curves.csv",
    index=False,
)


print()
print(
    "============================================================"
)
print(
    "BREAK-EVEN SUMMARY"
)
print(
    "============================================================"
)

print(
    summary.to_string(
        index=False
    )
)


# Compact manuscript-oriented AnyGold@3 table
primary = (
    summary[
        summary["metric"]
        == "AnyGold@3"
    ]
    .copy()
)

primary.to_csv(
    OUT
    / "metadata_reliability_anygold3.csv",
    index=False,
)


metadata = {
    "analysis":
        "metadata reliability operating-envelope analysis",
    "input":
        str(INPUT),
    "assumption_backward":
        (
            "When metadata is wrong, the supplied current lecture "
            "is exactly one lecture earlier."
        ),
    "assumption_forward":
        (
            "When metadata is wrong, the supplied current lecture "
            "is exactly one lecture later."
        ),
    "assumption_symmetric":
        (
            "For interior lectures only, wrong metadata is equally "
            "likely to be one lecture earlier or one lecture later."
        ),
    "important_scope":
        (
            "This analysis uses the fixed candidate pool from the "
            "metadata stress experiment. It characterizes reranker "
            "operating conditions, not end-to-end eligibility errors."
        ),
    "bootstrap_unit":
        "true target lecture",
    "bootstrap_replicates":
        N_BOOT,
    "seed":
        SEED,
}

(
    OUT
    / "METADATA_RELIABILITY_METADATA.json"
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
    "METADATA-RELIABILITY ANALYSIS COMPLETE"
)
print(
    "============================================================"
)
