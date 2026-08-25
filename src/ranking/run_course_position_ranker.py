#!/usr/bin/env python3
"""Leakage-free course-position-aware ranking.

This script evaluates two ranking systems on the position-eligible RRF
candidate pool:

1. A coarse lecture prior added to reciprocal-rank-fusion score.
2. A logistic ranker using content-ranking features together with only
   coarse lecture-position signals.

The valid feature set deliberately excludes:

- signed exact position distance
- absolute exact position distance
- within-lecture target-slide distance

The benchmark current position is anchored to the target instructional
location, so exact slide-position distance would reveal the target.

Cross-validation uses five-fold GroupKFold grouped by target lecture.
Gold membership is used only as a training label. Evaluation is performed
on the reranked candidate pool itself.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sentence_transformers import SentenceTransformer

from src.common.retrieval_primitives import (
    tokenize,
    doc_text,
    simple_bm25_scores,
    rrf,
)


ANSWERABLE_TYPES = {
    "evidence_cited_qa",
    "slide_local_factual_qa",
    "neighbor_slide_conceptual_qa",
}

KS = [1, 3, 5, 10, 20, 50]
SEED = 20260822
POOL_SIZE = 50
WEIGHT_GRID = [0.0, 0.05, 0.1, 0.2, 0.4, 0.8, 1.6]

SAFE_LOGISTIC_FEATURES = [
    "rrf_rr",
    "rrf_rank",
    "bm25_rr",
    "bm25_score",
    "tfidf_rr",
    "bge_sim",
    "same_lecture",
    "lecture_distance",
]

FORBIDDEN_POSITION_FEATURES = {
    "signed_position_distance",
    "abs_position_distance",
    "slide_distance_same_lecture",
}


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[2]

    parser = argparse.ArgumentParser(
        description="Leakage-free course-position ranking evaluation."
    )

    parser.add_argument(
        "--corpus",
        type=Path,
        default=root / "data/corpus/slide_corpus_final.jsonl",
    )

    parser.add_argument(
        "--benchmark",
        type=Path,
        default=root / "data/benchmark/core_qa_benchmark.jsonl",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root / "artifacts/course_position_ranking",
    )

    parser.add_argument(
        "--embedding-model",
        default="BAAI/bge-base-en-v1.5",
    )

    parser.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        default="cpu",
    )

    return parser.parse_args()


def load_corpus(path: Path):
    corpus = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    for row in corpus:
        row["lecture_id"] = int(row["lecture_id"])
        row["slide_id"] = int(row["slide_id"])

    ordered = sorted(
        corpus,
        key=lambda row: (row["lecture_id"], row["slide_id"]),
    )

    for index, row in enumerate(ordered):
        row["course_position"] = index

    return ordered


def load_tasks(path: Path, ordered):
    position_of = {
        row["doc_id"]: row["course_position"]
        for row in ordered
    }

    lecture_of = {
        row["doc_id"]: row["lecture_id"]
        for row in ordered
    }

    slide_of = {
        row["doc_id"]: row["slide_id"]
        for row in ordered
    }

    benchmark = pd.read_json(path, lines=True)

    benchmark = benchmark[
        benchmark["task_type"].isin(ANSWERABLE_TYPES)
    ]

    tasks = []

    for _, row in benchmark.iterrows():
        target = str(row.target_doc_id)

        if target not in position_of:
            continue

        tasks.append(
            {
                "task_id": str(row.task_id),
                "task_type": row.task_type,
                "question": row.question,
                "target": target,
                "gold": set(
                    str(x)
                    for x in row.evidence_doc_ids
                ),
                "target_position": position_of[target],
                "target_lecture": lecture_of[target],
                "target_slide": slide_of[target],
            }
        )

    tasks = sorted(
        tasks,
        key=lambda row: row["task_id"],
    )

    if len(tasks) != 600:
        raise RuntimeError(
            f"Expected 600 answerable tasks, found {len(tasks)}"
        )

    return tasks


def build_candidate_features(
    ordered,
    tasks,
    embedding_model: str,
    device: str,
):
    doc_ids = [
        row["doc_id"]
        for row in ordered
    ]

    lectures = np.array(
        [
            row["lecture_id"]
            for row in ordered
        ]
    )

    lecture_of = {
        row["doc_id"]: row["lecture_id"]
        for row in ordered
    }

    document_tokens = [
        tokenize(doc_text(row))
        for row in ordered
    ]

    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 3),
        max_df=0.85,
    )

    tfidf_matrix = vectorizer.fit_transform(
        [
            doc_text(row)
            for row in ordered
        ]
    )

    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA requested but unavailable."
        )

    encoder = SentenceTransformer(
        embedding_model,
        device=device,
    )

    document_embeddings = encoder.encode(
        [
            row.get("text", "")
            for row in ordered
        ],
        batch_size=32,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    )

    question_embeddings = encoder.encode(
        [
            task["question"]
            for task in tasks
        ],
        batch_size=32,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    rows = []
    base_ranking = {}

    for query_index, task in enumerate(tasks):

        eligible = {
            i
            for i, row in enumerate(ordered)
            if row["course_position"] <= task["target_position"]
        }

        bm25_scores = simple_bm25_scores(
            tokenize(task["question"]),
            document_tokens,
        )

        tfidf_scores = (
            tfidf_matrix
            @ vectorizer.transform([task["question"]]).T
        ).toarray().ravel()

        dense_scores = (
            document_embeddings
            @ question_embeddings[query_index]
        )

        def top(scores, k=POOL_SIZE):
            order = np.argsort(
                -np.asarray(scores),
                kind="stable",
            )

            return [
                int(i)
                for i in order
                if i in eligible
            ][:k]

        bm25_ranking = top(bm25_scores)
        tfidf_ranking = top(tfidf_scores)

        rrf_full = [
            i
            for i in rrf(
                [
                    bm25_ranking,
                    tfidf_ranking,
                ]
            )
            if i in eligible
        ]

        base_ranking[task["task_id"]] = rrf_full

        pool = rrf_full[:POOL_SIZE]

        rrf_position = {
            doc: rank
            for rank, doc in enumerate(rrf_full, 1)
        }

        bm25_position = {
            doc: rank
            for rank, doc in enumerate(bm25_ranking, 1)
        }

        tfidf_position = {
            doc: rank
            for rank, doc in enumerate(tfidf_ranking, 1)
        }

        for doc_index in pool:

            rows.append(
                {
                    "task_id": task["task_id"],
                    "task_type": task["task_type"],
                    "target_lecture": task["target_lecture"],
                    "doc": doc_ids[doc_index],
                    "rrf_rank":
                        rrf_position.get(doc_index, 999),
                    "rrf_rr":
                        1.0 / rrf_position.get(doc_index, 999),
                    "bm25_rank":
                        bm25_position.get(doc_index, 999),
                    "bm25_rr":
                        1.0 / bm25_position.get(doc_index, 999),
                    "bm25_score":
                        float(bm25_scores[doc_index]),
                    "tfidf_rank":
                        tfidf_position.get(doc_index, 999),
                    "tfidf_rr":
                        1.0 / tfidf_position.get(doc_index, 999),
                    "bge_sim":
                        float(dense_scores[doc_index]),
                    "same_lecture":
                        int(
                            lectures[doc_index]
                            == task["target_lecture"]
                        ),
                    "lecture_distance":
                        abs(
                            int(lectures[doc_index])
                            - int(task["target_lecture"])
                        ),
                    "is_gold":
                        int(
                            doc_ids[doc_index]
                            in task["gold"]
                        ),
                    "is_target":
                        int(
                            doc_ids[doc_index]
                            == task["target"]
                        ),
                }
            )

    features = pd.DataFrame(rows)

    return (
        features,
        base_ranking,
        doc_ids,
        lecture_of,
    )


def evaluate(
    name,
    order_map,
    tasks,
    lecture_of,
):
    k_rows = []
    mrr_rows = []

    for task in tasks:
        ranked = order_map[task["task_id"]]

        first_gold = next(
            (
                i + 1
                for i, doc in enumerate(ranked)
                if doc in task["gold"]
            ),
            None,
        )

        first_target = next(
            (
                i + 1
                for i, doc in enumerate(ranked)
                if doc == task["target"]
            ),
            None,
        )

        mrr_rows.append(
            {
                "method": name,
                "task_id": task["task_id"],
                "first_gold_rank": first_gold,
                "first_target_rank": first_target,
                "rr_any_gold":
                    1.0 / first_gold
                    if first_gold
                    else 0.0,
                "rr_target":
                    1.0 / first_target
                    if first_target
                    else 0.0,
            }
        )

        for k in KS:
            top = set(ranked[:k])

            k_rows.append(
                {
                    "method": name,
                    "k": k,
                    "task_id": task["task_id"],
                    "any_gold":
                        int(
                            bool(
                                top & task["gold"]
                            )
                        ),
                    "target_slide":
                        int(
                            task["target"]
                            in top
                        ),
                    "target_lecture":
                        int(
                            any(
                                lecture_of[doc]
                                == task["target_lecture"]
                                for doc in ranked[:k]
                            )
                        ),
                }
            )

    return (
        pd.DataFrame(k_rows),
        pd.DataFrame(mrr_rows),
    )


def main():
    args = parse_args()

    if (
        set(SAFE_LOGISTIC_FEATURES)
        & FORBIDDEN_POSITION_FEATURES
    ):
        raise RuntimeError(
            "Forbidden exact-position feature entered valid feature set."
        )

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    ordered = load_corpus(args.corpus)

    tasks = load_tasks(
        args.benchmark,
        ordered,
    )

    (
        features,
        base_ranking,
        doc_ids,
        lecture_of,
    ) = build_candidate_features(
        ordered,
        tasks,
        args.embedding_model,
        args.device,
    )

    features.to_csv(
        args.output_dir
        / "candidate_features.csv",
        index=False,
    )

    task_by_id = {
        task["task_id"]: task
        for task in tasks
    }

    task_ids = np.array(
        sorted(task_by_id)
    )

    groups = (
        features
        .groupby("task_id")
        ["target_lecture"]
        .first()
    )

    group_values = np.array(
        [
            groups[task_id]
            for task_id in task_ids
        ]
    )

    cross_validation = GroupKFold(
        n_splits=5
    )

    folds = list(
        cross_validation.split(
            task_ids,
            groups=group_values,
        )
    )

    all_k = []
    all_mrr = []

    # ---------------------------------------------------------
    # RRF baseline
    # ---------------------------------------------------------

    baseline_order = {
        task["task_id"]: [
            doc_ids[i]
            for i in base_ranking[
                task["task_id"]
            ]
        ]
        for task in tasks
    }

    k_df, mrr_df = evaluate(
        "rrf_baseline",
        baseline_order,
        tasks,
        lecture_of,
    )

    all_k.append(k_df)
    all_mrr.append(mrr_df)

    # ---------------------------------------------------------
    # 10A: coarse lecture prior
    # ---------------------------------------------------------

    def prior_score(subset, weight):
        return (
            subset["rrf_rr"].to_numpy(float)
            + weight
            * (
                subset["same_lecture"].to_numpy(float)
                - 0.01
                * subset["lecture_distance"].to_numpy(float)
            )
        )

    prior_order = {}
    selected_weights = []

    for fold_index, (train, test) in enumerate(
        folds,
        start=1,
    ):

        best_weight = None
        best_training_hits = -1

        for weight in WEIGHT_GRID:

            hits = 0

            for task_id in task_ids[train]:

                subset = features[
                    features["task_id"]
                    == task_id
                ]

                scores = prior_score(
                    subset,
                    weight,
                )

                ranked = subset["doc"].to_numpy()[
                    np.argsort(
                        -scores,
                        kind="stable",
                    )
                ]

                hits += int(
                    bool(
                        set(ranked[:3])
                        & task_by_id[task_id]["gold"]
                    )
                )

            if hits > best_training_hits:
                best_weight = weight
                best_training_hits = hits

        selected_weights.append(
            {
                "fold": fold_index,
                "weight": best_weight,
                "training_hits_at_3":
                    best_training_hits,
            }
        )

        for task_id in task_ids[test]:

            subset = features[
                features["task_id"]
                == task_id
            ]

            scores = prior_score(
                subset,
                best_weight,
            )

            prior_order[task_id] = list(
                subset["doc"].to_numpy()[
                    np.argsort(
                        -scores,
                        kind="stable",
                    )
                ]
            )

    k_df, mrr_df = evaluate(
        "10A_LEAKFREE_rrf_plus_coarse_lecture_only",
        prior_order,
        tasks,
        lecture_of,
    )

    all_k.append(k_df)
    all_mrr.append(mrr_df)

    # ---------------------------------------------------------
    # 10B: leakage-free logistic ranker
    # ---------------------------------------------------------

    logistic_order = {}

    for train, test in folds:

        train_ids = set(
            task_ids[train]
        )

        test_ids = set(
            task_ids[test]
        )

        train_rows = features[
            features["task_id"]
            .isin(train_ids)
        ]

        test_rows = features[
            features["task_id"]
            .isin(test_ids)
        ].copy()

        model = Pipeline(
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

        model.fit(
            train_rows[
                SAFE_LOGISTIC_FEATURES
            ].to_numpy(float),
            train_rows[
                "is_gold"
            ].to_numpy(),
        )

        test_rows["score"] = (
            model.predict_proba(
                test_rows[
                    SAFE_LOGISTIC_FEATURES
                ].to_numpy(float)
            )[:, 1]
        )

        for task_id, group in test_rows.groupby(
            "task_id",
            sort=True,
        ):

            logistic_order[str(task_id)] = (
                group
                .sort_values(
                    "score",
                    ascending=False,
                )
                ["doc"]
                .astype(str)
                .tolist()
            )

    k_df, mrr_df = evaluate(
        "10B_LEAKFREE_logistic_no_position_distance",
        logistic_order,
        tasks,
        lecture_of,
    )

    all_k.append(k_df)
    all_mrr.append(mrr_df)

    # ---------------------------------------------------------
    # Aggregate
    # ---------------------------------------------------------

    all_k_df = pd.concat(
        all_k,
        ignore_index=True,
    )

    all_mrr_df = pd.concat(
        all_mrr,
        ignore_index=True,
    )

    k_summary = (
        all_k_df
        .groupby(
            ["method", "k"],
            as_index=False,
        )
        .agg(
            N=("task_id", "nunique"),
            AnyGoldRecall=("any_gold", "mean"),
            TargetSlideRecall=("target_slide", "mean"),
            TargetLectureRecall=("target_lecture", "mean"),
        )
    )

    mrr_summary = (
        all_mrr_df
        .groupby(
            "method",
            as_index=False,
        )
        .agg(
            N=("task_id", "nunique"),
            MRRAnyGold=("rr_any_gold", "mean"),
            MRRTargetSlide=("rr_target", "mean"),
            MedianFirstGoldRank=(
                "first_gold_rank",
                "median",
            ),
        )
    )

    k_summary.to_csv(
        args.output_dir
        / "course_position_ranker_k_sweep.csv",
        index=False,
    )

    mrr_summary.to_csv(
        args.output_dir
        / "course_position_ranker_mrr.csv",
        index=False,
    )

    pd.DataFrame(
        selected_weights
    ).to_csv(
        args.output_dir
        / "coarse_prior_selected_weights.csv",
        index=False,
    )

    fold_rows = []

    for fold_index, (train, test) in enumerate(
        folds,
        start=1,
    ):

        train_lectures = set(
            group_values[train]
        )

        test_lectures = set(
            group_values[test]
        )

        fold_rows.append(
            {
                "fold": fold_index,
                "n_train_tasks":
                    len(train),
                "n_test_tasks":
                    len(test),
                "n_train_lectures":
                    len(train_lectures),
                "n_test_lectures":
                    len(test_lectures),
                "disjoint_target_lectures":
                    len(
                        train_lectures
                        & test_lectures
                    ) == 0,
            }
        )

    pd.DataFrame(
        fold_rows
    ).to_csv(
        args.output_dir
        / "grouped_cv_folds.csv",
        index=False,
    )

    print()
    print(
        k_summary[
            k_summary["k"].isin(
                [1, 3, 10, 50]
            )
        ]
        .pivot_table(
            index="method",
            columns="k",
            values="AnyGoldRecall",
        )
        .round(4)
        .to_string()
    )

    print()
    print(
        mrr_summary
        .round(4)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
