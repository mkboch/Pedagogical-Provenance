#!/usr/bin/env python3
"""
Post-freeze scale and boundary audit for the PACER coarse lecture prior.

This utility consumes candidate_features.csv produced by the public
course-position ranker, or the equivalent frozen local candidate-feature
artifact.

It does not alter the frozen ranking protocol. It evaluates:

1. The complete original weight grid.
2. Extended diagnostic weights 3.2 and 6.4.
3. The analytic strict lecture-block threshold.
4. Same-lecture candidate coverage.
5. Original and extended GroupKFold weight selection.
6. A parameter-free same-lecture-first then RRF comparator.

Generated audit output belongs under artifacts and is not tracked by Git.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold


ORIGINAL_GRID = [0.0, 0.05, 0.10, 0.20, 0.40, 0.80, 1.60]
EXTENDED_GRID = ORIGINAL_GRID + [3.20, 6.40]
K = 3

REQUIRED = {
    "task_id",
    "target_lecture",
    "doc",
    "rrf_rank",
    "rrf_rr",
    "same_lecture",
    "lecture_distance",
    "is_gold",
    "is_target",
}


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[2]

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--candidate-features",
        type=Path,
        default=root / "artifacts/course_position_ranking/candidate_features.csv",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=root / "artifacts/audits/lecture_prior_boundary_audit.json",
    )

    return parser.parse_args()


def ranked_with_weight(group: pd.DataFrame, weight: float) -> pd.DataFrame:
    scores = (
        group["rrf_rr"].to_numpy(float)
        + weight
        * (
            group["same_lecture"].to_numpy(float)
            - 0.01 * group["lecture_distance"].to_numpy(float)
        )
    )

    order = np.argsort(
        -scores,
        kind="stable",
    )

    return group.iloc[order]


def ranked_parameter_free(group: pd.DataFrame) -> pd.DataFrame:
    return group.sort_values(
        ["same_lecture", "rrf_rank"],
        ascending=[False, True],
        kind="stable",
    )


def task_outcome(ranked: pd.DataFrame):
    top = ranked.iloc[:K]

    any_gold = int(
        top["is_gold"].astype(int).any()
    )

    target = int(
        top["is_target"].astype(int).any()
    )

    docs = tuple(
        top["doc"].astype(str)
    )

    return any_gold, target, docs


def evaluate_weight(
    features: pd.DataFrame,
    weight: float,
    task_ids=None,
):
    any_hits = 0
    target_hits = 0
    docs = {}
    outcomes = {}

    for task_id, group in features.groupby(
        "task_id",
        sort=True,
    ):
        task_id = str(task_id)

        if (
            task_ids is not None
            and task_id not in task_ids
        ):
            continue

        ranked = ranked_with_weight(
            group,
            weight,
        )

        any_gold, target, top_docs = task_outcome(
            ranked
        )

        any_hits += any_gold
        target_hits += target

        docs[task_id] = top_docs
        outcomes[task_id] = (
            any_gold,
            target,
        )

    return (
        any_hits,
        target_hits,
        docs,
        outcomes,
    )


def evaluate_parameter_free(features: pd.DataFrame):
    any_hits = 0
    target_hits = 0
    docs = {}
    outcomes = {}

    for task_id, group in features.groupby(
        "task_id",
        sort=True,
    ):
        task_id = str(task_id)

        ranked = ranked_parameter_free(
            group
        )

        any_gold, target, top_docs = task_outcome(
            ranked
        )

        any_hits += any_gold
        target_hits += target

        docs[task_id] = top_docs
        outcomes[task_id] = (
            any_gold,
            target,
        )

    return (
        any_hits,
        target_hits,
        docs,
        outcomes,
    )


def fold_audit(
    features: pd.DataFrame,
    grid: list[float],
):
    task_ids = np.array(
        sorted(
            features["task_id"]
            .astype(str)
            .unique()
        )
    )

    target_lecture = (
        features
        .assign(
            task_id=features["task_id"].astype(str)
        )
        .groupby("task_id")["target_lecture"]
        .first()
    )

    groups = np.array(
        [
            target_lecture[task_id]
            for task_id in task_ids
        ]
    )

    cross_validation = GroupKFold(
        n_splits=5
    )

    folds = list(
        cross_validation.split(
            task_ids,
            groups=groups,
        )
    )

    fold_rows = []
    oof_any = {}
    oof_target = {}

    for fold_index, (train, test) in enumerate(
        folds,
        start=1,
    ):
        train_ids = set(
            task_ids[train]
        )

        test_ids = set(
            task_ids[test]
        )

        training_curve = {}

        best_weight = None
        best_hits = -1

        for weight in grid:
            hits, _, _, _ = evaluate_weight(
                features,
                weight,
                task_ids=train_ids,
            )

            training_curve[
                f"{weight:g}"
            ] = int(hits)

            # Strict improvement reproduces the released
            # implementation. Ties therefore retain the
            # earlier and smaller weight.
            if hits > best_hits:
                best_weight = weight
                best_hits = hits

        (
            test_any,
            test_target,
            _,
            test_outcomes,
        ) = evaluate_weight(
            features,
            best_weight,
            task_ids=test_ids,
        )

        for task_id, outcome in test_outcomes.items():
            oof_any[task_id] = outcome[0]
            oof_target[task_id] = outcome[1]

        fold_rows.append(
            {
                "fold": fold_index,
                "n_train_tasks": int(
                    len(train)
                ),
                "n_test_tasks": int(
                    len(test)
                ),
                "held_out_lectures": sorted(
                    {
                        int(x)
                        for x in groups[test]
                    }
                ),
                "training_anygold_at_3":
                    training_curve,
                "selected_weight":
                    float(best_weight),
                "test_anygold_hits_at_3":
                    int(test_any),
                "test_target_hits_at_3":
                    int(test_target),
            }
        )

    return {
        "folds": fold_rows,
        "selected_weights": [
            row["selected_weight"]
            for row in fold_rows
        ],
        "oof_anygold_hits_at_3":
            int(sum(oof_any.values())),
        "oof_target_hits_at_3":
            int(sum(oof_target.values())),
    }


def main():
    args = parse_args()

    features = pd.read_csv(
        args.candidate_features
    )

    missing = sorted(
        REQUIRED - set(features.columns)
    )

    if missing:
        raise SystemExit(
            f"Missing required columns: {missing}"
        )

    features = features.copy()

    features["task_id"] = (
        features["task_id"]
        .astype(str)
    )

    n_tasks = int(
        features["task_id"].nunique()
    )

    if n_tasks == 0:
        raise SystemExit(
            "No tasks found"
        )

    rr_min = float(
        features["rrf_rr"].min()
    )

    rr_max = float(
        features["rrf_rr"].max()
    )

    rr_spread = (
        rr_max - rr_min
    )

    different_lecture = features[
        features["same_lecture"]
        .astype(int)
        .eq(0)
    ]

    if len(different_lecture):
        min_different_lecture_distance = float(
            different_lecture[
                "lecture_distance"
            ]
            .astype(float)
            .min()
        )
    else:
        min_different_lecture_distance = 1.0

    strict_threshold = (
        rr_spread
        /
        (
            1.0
            + 0.01
            * min_different_lecture_distance
        )
    )

    same_counts = (
        features
        .groupby("task_id")["same_lecture"]
        .sum()
        .astype(int)
    )

    n_three_or_more = int(
        (same_counts >= 3).sum()
    )

    n_fewer_than_three = int(
        (same_counts < 3).sum()
    )

    full_grid = {}

    for weight in EXTENDED_GRID:
        any_hits, target_hits, _, _ = (
            evaluate_weight(
                features,
                weight,
            )
        )

        full_grid[
            f"{weight:g}"
        ] = {
            "AnyGold_hits_at_3":
                int(any_hits),
            "AnyGold_at_3":
                float(
                    any_hits / n_tasks
                ),
            "TargetSlide_hits_at_3":
                int(target_hits),
            "TargetSlide_at_3":
                float(
                    target_hits / n_tasks
                ),
        }

    low_same_ids = set(
        same_counts[
            same_counts < 3
        ]
        .index
        .astype(str)
    )

    low_same_grid = {}

    for weight in EXTENDED_GRID:
        any_hits, target_hits, _, _ = (
            evaluate_weight(
                features,
                weight,
                task_ids=low_same_ids,
            )
        )

        low_same_grid[
            f"{weight:g}"
        ] = {
            "AnyGold_hits_at_3":
                int(any_hits),
            "TargetSlide_hits_at_3":
                int(target_hits),
        }

    (
        frozen_any,
        frozen_target,
        frozen_docs,
        frozen_outcomes,
    ) = evaluate_weight(
        features,
        1.6,
    )

    (
        parameter_any,
        parameter_target,
        parameter_docs,
        parameter_outcomes,
    ) = evaluate_parameter_free(
        features
    )

    exact_top3_differences = sum(
        frozen_docs[task_id]
        != parameter_docs[task_id]
        for task_id in frozen_docs
    )

    any_outcome_differences = sum(
        frozen_outcomes[task_id][0]
        != parameter_outcomes[task_id][0]
        for task_id in frozen_outcomes
    )

    target_outcome_differences = sum(
        frozen_outcomes[task_id][1]
        != parameter_outcomes[task_id][1]
        for task_id in frozen_outcomes
    )

    original_folds = fold_audit(
        features,
        ORIGINAL_GRID,
    )

    extended_folds = fold_audit(
        features,
        EXTENDED_GRID,
    )

    report = {
        "candidate_feature_file":
            str(args.candidate_features),
        "n_candidate_rows":
            int(len(features)),
        "n_tasks":
            n_tasks,
        "k":
            K,
        "score_definition":
            "rrf_rr + w * (same_lecture - 0.01 * lecture_distance)",
        "rrf_rr_min":
            rr_min,
        "rrf_rr_max":
            rr_max,
        "rrf_rr_spread":
            rr_spread,
        "minimum_different_lecture_distance":
            min_different_lecture_distance,
        "strict_same_lecture_block_condition": {
            "form":
                "w > spread / (1 + 0.01 * minimum_different_lecture_distance)",
            "threshold":
                float(strict_threshold),
        },
        "same_lecture_candidate_coverage": {
            "tasks_with_at_least_3_same_lecture_candidates":
                n_three_or_more,
            "tasks_with_fewer_than_3_same_lecture_candidates":
                n_fewer_than_three,
            "fraction_with_at_least_3":
                float(
                    n_three_or_more
                    / n_tasks
                ),
            "interpretation":
                "For tasks with at least three same-lecture candidates, "
                "the top three are invariant for every weight above the "
                "strict lecture-block threshold because the common positive "
                "weight term cancels within the same-lecture block.",
        },
        "full_weight_response":
            full_grid,
        "fewer_than_3_same_lecture_weight_response":
            low_same_grid,
        "original_grid_groupkfold":
            original_folds,
        "extended_grid_groupkfold":
            extended_folds,
        "parameter_free_same_lecture_first_then_rrf": {
            "AnyGold_hits_at_3":
                int(parameter_any),
            "AnyGold_at_3":
                float(
                    parameter_any
                    / n_tasks
                ),
            "TargetSlide_hits_at_3":
                int(parameter_target),
            "TargetSlide_at_3":
                float(
                    parameter_target
                    / n_tasks
                ),
            "frozen_w_1_6_AnyGold_hits_at_3":
                int(frozen_any),
            "frozen_w_1_6_TargetSlide_hits_at_3":
                int(frozen_target),
            "exact_top3_list_differences_vs_w_1_6":
                int(
                    exact_top3_differences
                ),
            "AnyGold_task_outcome_differences_vs_w_1_6":
                int(
                    any_outcome_differences
                ),
            "TargetSlide_task_outcome_differences_vs_w_1_6":
                int(
                    target_outcome_differences
                ),
        },
    }

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    args.output.write_text(
        json.dumps(
            report,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print(
        json.dumps(
            report,
            indent=2,
        )
    )

    print(
        f"\nWrote: {args.output}"
    )


if __name__ == "__main__":
    main()
