#!/usr/bin/env python3

from pathlib import Path
from collections import Counter
import json
import math
import os
import re
import warnings

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "artifacts/trigger_robustness_audit"
OUT.mkdir(parents=True, exist_ok=True)

SEED = 20260817

LABEL_MAP = {
    "evidence_cited_qa": "answerable",
    "slide_local_factual_qa": "answerable",
    "neighbor_slide_conceptual_qa": "answerable",
    "out_of_scope_abstention_qa": "clinical_out_of_scope",
    "lecture_sequence_violation_probe": "sequence_sensitive",
}

LABELS = [
    "answerable",
    "clinical_out_of_scope",
    "sequence_sensitive",
]

REPORT = []


def section(title):
    REPORT.append("")
    REPORT.append("=" * 100)
    REPORT.append(title)
    REPORT.append("=" * 100)


def canonical_doc_id(value):
    s = str(value).strip()

    m = re.search(
        r"lecture[_\-\s]*(\d+)[_\-\s]*slide[_\-\s]*(\d+)",
        s,
        flags=re.I,
    )

    if m:
        return (
            f"lecture_{int(m.group(1)):02d}_"
            f"slide_{int(m.group(2)):03d}"
        )

    return s.lower()


def parse_position(value):
    s = canonical_doc_id(value)

    m = re.search(
        r"lecture_(\d+)_slide_(\d+)",
        s
    )

    if not m:
        return None

    return int(m.group(1)), int(m.group(2))


# =============================================================================
# PRIMARY BENCHMARK
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
            "FAILED: primary detail file not found"
        )

    DETAIL = matches[0]

raw = pd.read_csv(DETAIL)

required = [
    "task_id",
    "task_type",
    "question",
    "target_doc_id",
]

missing = [
    x for x in required
    if x not in raw.columns
]

if missing:
    raise SystemExit(
        f"FAILED: primary data missing {missing}"
    )

df = (
    raw[required]
    .drop_duplicates("task_id")
    .copy()
)

df = df[
    df["task_type"].isin(LABEL_MAP)
].copy()

df["label"] = (
    df["task_type"]
    .map(LABEL_MAP)
)

positions = (
    df["target_doc_id"]
    .map(parse_position)
)

df["lecture"] = positions.map(
    lambda x: x[0] if x else np.nan
)

df["slide"] = positions.map(
    lambda x: x[1] if x else np.nan
)

df = df.dropna(
    subset=[
        "lecture",
        "slide",
        "question",
        "label",
    ]
).copy()

df["lecture"] = (
    df["lecture"]
    .astype(int)
)

df["slide"] = (
    df["slide"]
    .astype(int)
)

df["question"] = (
    df["question"]
    .astype(str)
)

if len(df) != 1000:
    raise SystemExit(
        f"FAILED: expected 1000 tasks, found {len(df)}"
    )


# =============================================================================
# CUE STRIPPING
#
# Purpose:
# remove benchmark-family wrapper phrases while preserving the underlying
# student request/concept as much as possible.
#
# This is NOT a replacement benchmark.
# It is a robustness diagnostic.
# =============================================================================

def clean_spaces(s):
    s = re.sub(
        r"[ \t]+",
        " ",
        s,
    )

    s = re.sub(
        r"\s+([,.;:?!])",
        r"\1",
        s,
    )

    s = re.sub(
        r"\n+",
        " ",
        s,
    )

    return s.strip(
        " ,;:-"
    )


def strip_benchmark_cues(question, task_type):
    q = str(question).strip()

    # ---------------------------------------------------------
    # Remove generic citation wrapper from evidence-cited QA.
    # Example:
    # "Answer briefly and cite the supporting slide ID. What does..."
    # ---------------------------------------------------------
    q = re.sub(
        r"^\s*Answer\s+briefly\s+and\s+cite\s+the\s+supporting\s+slide\s+ID\.\s*",
        "",
        q,
        flags=re.I,
    )

    q = re.sub(
        r"^\s*Please\s+cite\s+the\s+supporting\s+slide\s*(?:ID)?\.?\s*",
        "",
        q,
        flags=re.I,
    )

    # ---------------------------------------------------------
    # Sequence probe:
    # remove explicit benchmark position wrapper while retaining
    # the requested concept.
    #
    # Original form:
    # At Lecture 4, Slide 20, explain ultrasound using only
    # concepts introduced up to that point in the course.
    #
    # Diagnostic form:
    # explain ultrasound
    # ---------------------------------------------------------
    if task_type == "lecture_sequence_violation_probe":

        q = re.sub(
            r"^\s*At\s+Lecture\s+\d+\s*,?\s*Slide\s+\d+\s*,?\s*",
            "",
            q,
            flags=re.I,
        )

        q = re.sub(
            r"\s+using\s+only\s+concepts\s+introduced\s+up\s+to\s+that\s+point\s+in\s+the\s+course\s*\.?\s*$",
            "",
            q,
            flags=re.I,
        )

        q = re.sub(
            r"\s+using\s+only\s+material\s+available\s+up\s+to\s+that\s+point\s*\.?\s*$",
            "",
            q,
            flags=re.I,
        )

        q = re.sub(
            r"\s+based\s+only\s+on\s+what\s+has\s+been\s+introduced\s+so\s+far\s*\.?\s*$",
            "",
            q,
            flags=re.I,
        )

    # ---------------------------------------------------------
    # Clinical trap:
    # remove course-position wrapper but KEEP clinical request.
    #
    # Original:
    # Considering Lecture X, Slide Y about foo, [clinical request]
    #
    # Diagnostic:
    # [clinical request]
    # ---------------------------------------------------------
    if task_type == "out_of_scope_abstention_qa":

        q = re.sub(
            r"^\s*Considering\s+Lecture\s+\d+\s*,?\s*Slide\s+\d+"
            r"(?:\s+about\s+[^,?]+)?\s*,?\s*",
            "",
            q,
            flags=re.I,
        )

    # ---------------------------------------------------------
    # Other generic position wrappers.
    # ---------------------------------------------------------
    q = re.sub(
        r"^\s*(?:At|In|From)\s+Lecture\s+\d+\s*,?\s*Slide\s+\d+\s*,?\s*",
        "",
        q,
        flags=re.I,
    )

    return clean_spaces(q)


df["question_cuestripped"] = [
    strip_benchmark_cues(
        q,
        tt,
    )
    for q, tt in zip(
        df["question"],
        df["task_type"],
    )
]

df["question_changed"] = (
    df["question"].str.strip()
    !=
    df["question_cuestripped"].str.strip()
).astype(int)

df.to_csv(
    OUT / "primary_questions_original_and_cuestripped.csv",
    index=False,
)


# =============================================================================
# SHOW EXAMPLES
# =============================================================================

section("1. CUE-STRIPPING AUDIT")

REPORT.append(
    f"Primary benchmark: {DETAIL}"
)

REPORT.append(
    f"Tasks: {len(df)}"
)

REPORT.append("")
REPORT.append(
    "Questions changed by family:"
)

changed = (
    df.groupby("task_type")
    .agg(
        N=("task_id", "count"),
        Changed=("question_changed", "sum"),
    )
    .reset_index()
)

changed["ChangedRate"] = (
    changed["Changed"]
    / changed["N"]
)

REPORT.append(
    changed.to_string(
        index=False
    )
)

REPORT.append("")
REPORT.append(
    "Examples:"
)

for tt in LABEL_MAP:

    subset = df[
        df["task_type"].eq(tt)
        & df["question_changed"].eq(1)
    ].head(3)

    if subset.empty:
        subset = df[
            df["task_type"].eq(tt)
        ].head(3)

    REPORT.append("")
    REPORT.append(
        f"[{tt}]"
    )

    for _, r in subset.iterrows():
        REPORT.append(
            "ORIGINAL : "
            + r["question"]
        )
        REPORT.append(
            "STRIPPED : "
            + r["question_cuestripped"]
        )
        REPORT.append("")


# =============================================================================
# FEATURE VIEWS
# =============================================================================

def position_string(row):
    return (
        f"COURSE_LECTURE_{int(row['lecture'])} "
        f"COURSE_SLIDE_{int(row['slide'])}"
    )


df["position_only"] = (
    df.apply(
        position_string,
        axis=1,
    )
)

df["original_question_only"] = (
    df["question"]
)

df["original_question_position"] = (
    df["question"]
    + " "
    + df["position_only"]
)

df["stripped_question_only"] = (
    df["question_cuestripped"]
)

df["stripped_question_position"] = (
    df["question_cuestripped"]
    + " "
    + df["position_only"]
)


# =============================================================================
# CLASSIFIER
# =============================================================================

from sklearn.base import clone
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    classification_report,
)

def make_model():
    return Pipeline([
        (
            "tfidf",
            TfidfVectorizer(
                lowercase=True,
                ngram_range=(1, 2),
                min_df=2,
                max_df=0.98,
                sublinear_tf=True,
                max_features=30000,
            ),
        ),
        (
            "clf",
            LogisticRegression(
                C=1.0,
                max_iter=3000,
                class_weight="balanced",
                random_state=SEED,
            ),
        ),
    ])


# =============================================================================
# MANUAL GROUPED OOF
#
# Needed because some experiments intentionally train on original text
# but evaluate the same held-out examples after cue stripping.
# =============================================================================

y = (
    df["label"]
    .to_numpy()
)

groups = (
    df["lecture"]
    .to_numpy()
)

sgkf = StratifiedGroupKFold(
    n_splits=5,
    shuffle=True,
    random_state=SEED,
)


def evaluate(
    name,
    train_column,
    test_column,
):
    preds = np.empty(
        len(df),
        dtype=object,
    )

    fold_ids = np.zeros(
        len(df),
        dtype=int,
    )

    for fold, (tr, te) in enumerate(
        sgkf.split(
            df,
            y,
            groups,
        ),
        start=1,
    ):
        model = make_model()

        Xtr = (
            df.iloc[tr][train_column]
            .astype(str)
            .to_numpy()
        )

        Xte = (
            df.iloc[te][test_column]
            .astype(str)
            .to_numpy()
        )

        model.fit(
            Xtr,
            y[tr],
        )

        preds[te] = model.predict(
            Xte
        )

        fold_ids[te] = fold

    acc = accuracy_score(
        y,
        preds,
    )

    bacc = balanced_accuracy_score(
        y,
        preds,
    )

    macro = f1_score(
        y,
        preds,
        labels=LABELS,
        average="macro",
        zero_division=0,
    )

    weighted = f1_score(
        y,
        preds,
        labels=LABELS,
        average="weighted",
        zero_division=0,
    )

    answer_mask = (
        y == "answerable"
    )

    clinical_mask = (
        y == "clinical_out_of_scope"
    )

    sequence_mask = (
        y == "sequence_sensitive"
    )

    false_abstention = np.mean(
        preds[answer_mask]
        != "answerable"
    )

    clinical_miss = np.mean(
        preds[clinical_mask]
        != "clinical_out_of_scope"
    )

    sequence_miss = np.mean(
        preds[sequence_mask]
        != "sequence_sensitive"
    )

    summary = {
        "experiment": name,
        "train_view": train_column,
        "test_view": test_column,
        "N": len(df),
        "accuracy": acc,
        "balanced_accuracy": bacc,
        "macro_f1": macro,
        "weighted_f1": weighted,
        "false_abstention_rate": false_abstention,
        "clinical_miss_rate": clinical_miss,
        "sequence_miss_rate": sequence_miss,
    }

    pred_df = pd.DataFrame({
        "task_id":
            df["task_id"].astype(str),

        "task_type":
            df["task_type"].astype(str),

        "lecture":
            df["lecture"],

        "slide":
            df["slide"],

        "fold":
            fold_ids,

        "truth":
            y,

        "prediction":
            preds,

        "correct":
            (
                y == preds
            ).astype(int),

        "experiment":
            name,

        "original_question":
            df["question"],

        "stripped_question":
            df["question_cuestripped"],
    })

    pred_df.to_csv(
        OUT / f"{name}_predictions.csv",
        index=False,
    )

    cm = confusion_matrix(
        y,
        preds,
        labels=LABELS,
    )

    cm_df = pd.DataFrame(
        cm,
        index=[
            f"true_{x}"
            for x in LABELS
        ],
        columns=[
            f"pred_{x}"
            for x in LABELS
        ],
    )

    cm_df.to_csv(
        OUT / f"{name}_confusion_matrix.csv"
    )

    errors = pred_df[
        pred_df["correct"].eq(0)
    ].copy()

    errors.to_csv(
        OUT / f"{name}_errors.csv",
        index=False,
    )

    return (
        summary,
        cm_df,
        classification_report(
            y,
            preds,
            labels=LABELS,
            digits=4,
            zero_division=0,
        ),
    )


experiments = [
    (
        "E1_original_question_only",
        "original_question_only",
        "original_question_only",
    ),
    (
        "E2_position_only",
        "position_only",
        "position_only",
    ),
    (
        "E3_original_question_plus_position",
        "original_question_position",
        "original_question_position",
    ),
    (
        "E4_stripped_train_stripped_test_question_only",
        "stripped_question_only",
        "stripped_question_only",
    ),
    (
        "E5_stripped_train_stripped_test_plus_position",
        "stripped_question_position",
        "stripped_question_position",
    ),
    (
        "E6_original_train_stripped_test_question_only",
        "original_question_only",
        "stripped_question_only",
    ),
    (
        "E7_original_train_stripped_test_plus_position",
        "original_question_position",
        "stripped_question_position",
    ),
]

summary_rows = []

section("2. LECTURE-GROUPED ROBUSTNESS EXPERIMENTS")

for (
    name,
    train_col,
    test_col,
) in experiments:

    print(
        "Running",
        name,
    )

    summary, cm, report = evaluate(
        name,
        train_col,
        test_col,
    )

    summary_rows.append(
        summary
    )

    REPORT.append("")
    REPORT.append(
        f"[{name}]"
    )

    REPORT.append(
        json.dumps(
            summary,
            indent=2,
        )
    )

    REPORT.append("")
    REPORT.append(
        cm.to_string()
    )

    REPORT.append("")
    REPORT.append(
        report
    )


summary_df = pd.DataFrame(
    summary_rows
)

summary_df.to_csv(
    OUT / "trigger_robustness_summary.csv",
    index=False,
)


# =============================================================================
# TRIVIAL TEMPLATE-RULE BASELINE
# =============================================================================

section("3. TRIVIAL TEMPLATE-RULE BASELINE")

def template_rule(q):
    s = str(q).lower()

    if (
        "using only concepts introduced up to that point"
        in s
        or
        "introduced up to that point in the course"
        in s
    ):
        return "sequence_sensitive"

    clinical_terms = [
        "patient-specific",
        "medication dosage",
        "treatment planning",
        "treatment plan",
        "hospital protocol",
        "regulatory guidance",
        "clinical decision",
        "diagnose",
        "diagnosis",
        "prescribe",
    ]

    if any(
        x in s
        for x in clinical_terms
    ):
        return "clinical_out_of_scope"

    return "answerable"


rule_pred_original = np.array([
    template_rule(q)
    for q in df["question"]
])

rule_pred_stripped = np.array([
    template_rule(q)
    for q in df["question_cuestripped"]
])


def rule_stats(pred, name):
    return {
        "experiment": name,
        "N": len(y),
        "accuracy": accuracy_score(
            y,
            pred,
        ),
        "balanced_accuracy":
            balanced_accuracy_score(
                y,
                pred,
            ),
        "macro_f1":
            f1_score(
                y,
                pred,
                labels=LABELS,
                average="macro",
                zero_division=0,
            ),
        "false_abstention_rate":
            np.mean(
                pred[
                    y == "answerable"
                ]
                != "answerable"
            ),
        "clinical_miss_rate":
            np.mean(
                pred[
                    y
                    == "clinical_out_of_scope"
                ]
                != "clinical_out_of_scope"
            ),
        "sequence_miss_rate":
            np.mean(
                pred[
                    y
                    == "sequence_sensitive"
                ]
                != "sequence_sensitive"
            ),
    }


rule_original_stats = rule_stats(
    rule_pred_original,
    "template_rule_original",
)

rule_stripped_stats = rule_stats(
    rule_pred_stripped,
    "template_rule_stripped",
)

REPORT.append(
    json.dumps(
        rule_original_stats,
        indent=2,
    )
)

REPORT.append("")

REPORT.append(
    json.dumps(
        rule_stripped_stats,
        indent=2,
    )
)

pd.DataFrame([
    rule_original_stats,
    rule_stripped_stats,
]).to_csv(
    OUT / "template_rule_baseline.csv",
    index=False,
)


# =============================================================================
# EXTERNAL LPM INVENTORY
#
# Search likely benchmark/result files only, not the entire extracted
# raw LPM directory.
# =============================================================================

section("4. EXTERNAL LPM DATASET INVENTORY")

LPM_ROOT = (
    ROOT
    / "external_validation"
    / "lpm"
)

candidate_paths = []

preferred_dirs = [
    LPM_ROOT / "results_qwen3",
    LPM_ROOT / "medped_external_candidate",
    LPM_ROOT / "benchmark",
    LPM_ROOT / "benchmarks",
    LPM_ROOT / "tasks",
]

for directory in preferred_dirs:

    if not directory.exists():
        continue

    for p in directory.rglob("*"):

        if (
            p.is_file()
            and p.suffix.lower()
            in {
                ".csv",
                ".json",
                ".jsonl",
            }
        ):
            candidate_paths.append(
                p
            )

candidate_paths = sorted(
    set(candidate_paths)
)


def load_table(path):

    try:
        if path.suffix.lower() == ".csv":

            return pd.read_csv(
                path
            )

        if path.suffix.lower() == ".jsonl":

            rows = []

            with path.open(
                "r",
                encoding="utf-8",
                errors="ignore",
            ) as f:

                for line in f:
                    line = line.strip()

                    if not line:
                        continue

                    try:
                        rows.append(
                            json.loads(
                                line
                            )
                        )
                    except Exception:
                        pass

            return pd.DataFrame(
                rows
            )

        if path.suffix.lower() == ".json":

            obj = json.loads(
                path.read_text(
                    encoding="utf-8",
                    errors="ignore",
                )
            )

            if isinstance(
                obj,
                list,
            ):
                return pd.DataFrame(
                    obj
                )

            if isinstance(
                obj,
                dict,
            ):

                for key in [
                    "tasks",
                    "items",
                    "data",
                    "records",
                    "examples",
                ]:

                    if (
                        key in obj
                        and isinstance(
                            obj[key],
                            list,
                        )
                    ):
                        return pd.DataFrame(
                            obj[key]
                        )

    except Exception:
        return None

    return None


inventory = []

usable_tables = []

for p in candidate_paths:

    tab = load_table(
        p
    )

    if (
        tab is None
        or tab.empty
    ):
        continue

    lower_cols = {
        str(c).lower(): c
        for c in tab.columns
    }

    q_col = None
    t_col = None
    id_col = None

    for cand in [
        "question",
        "query",
    ]:
        if cand in lower_cols:
            q_col = lower_cols[cand]
            break

    for cand in [
        "task_type",
        "type",
    ]:
        if cand in lower_cols:
            t_col = lower_cols[cand]
            break

    for cand in [
        "task_id",
        "id",
    ]:
        if cand in lower_cols:
            id_col = lower_cols[cand]
            break

    row = {
        "path":
            str(
                p.relative_to(ROOT)
            ),

        "rows":
            len(tab),

        "has_question":
            int(
                q_col is not None
            ),

        "has_task_type":
            int(
                t_col is not None
            ),

        "has_task_id":
            int(
                id_col is not None
            ),
    }

    if t_col is not None:

        vc = (
            tab[t_col]
            .astype(str)
            .value_counts()
        )

        row[
            "known_task_rows"
        ] = int(
            tab[t_col]
            .astype(str)
            .isin(LABEL_MAP)
            .sum()
        )

        row[
            "task_counts"
        ] = json.dumps(
            vc.to_dict()
        )

    else:
        row[
            "known_task_rows"
        ] = 0

        row[
            "task_counts"
        ] = "{}"

    inventory.append(
        row
    )

    if (
        q_col is not None
        and t_col is not None
    ):
        usable_tables.append(
            (
                p,
                tab,
                q_col,
                t_col,
                id_col,
            )
        )


inventory_df = pd.DataFrame(
    inventory
)

if not inventory_df.empty:

    inventory_df = inventory_df.sort_values(
        [
            "known_task_rows",
            "rows",
        ],
        ascending=[
            False,
            True,
        ],
    )

inventory_df.to_csv(
    OUT / "lpm_candidate_inventory.csv",
    index=False,
)

REPORT.append(
    inventory_df.head(50).to_string(
        index=False
    )
)


# =============================================================================
# IDENTIFY BEST 150-160-ITEM LPM BENCHMARK
# =============================================================================

section("5. FULL LPM TRIGGER EVALUATION")

selected = None

# First priority:
# a file with approximately the reported 155 unique tasks and all
# three relevant trigger labels represented.
for (
    p,
    tab,
    q_col,
    t_col,
    id_col,
) in usable_tables:

    temp = tab[
        tab[t_col]
        .astype(str)
        .isin(LABEL_MAP)
    ].copy()

    if id_col is not None:
        temp = temp.drop_duplicates(
            id_col
        )
    else:
        temp = temp.drop_duplicates(
            q_col
        )

    labels_present = set(
        temp[t_col]
        .astype(str)
        .map(LABEL_MAP)
        .dropna()
    )

    if (
        150 <= len(temp) <= 160
        and
        set(LABELS).issubset(
            labels_present
        )
    ):
        selected = (
            p,
            temp,
            q_col,
            t_col,
            id_col,
        )
        break


# Second priority:
# closest-size candidate containing all 3 labels.
if selected is None:

    options = []

    for (
        p,
        tab,
        q_col,
        t_col,
        id_col,
    ) in usable_tables:

        temp = tab[
            tab[t_col]
            .astype(str)
            .isin(LABEL_MAP)
        ].copy()

        if id_col is not None:
            temp = temp.drop_duplicates(
                id_col
            )
        else:
            temp = temp.drop_duplicates(
                q_col
            )

        labels_present = set(
            temp[t_col]
            .astype(str)
            .map(LABEL_MAP)
            .dropna()
        )

        if set(LABELS).issubset(
            labels_present
        ):

            options.append(
                (
                    abs(
                        len(temp)
                        - 155
                    ),
                    p,
                    temp,
                    q_col,
                    t_col,
                    id_col,
                )
            )

    if options:

        options.sort(
            key=lambda x: x[0]
        )

        _, p, temp, q_col, t_col, id_col = options[0]

        selected = (
            p,
            temp,
            q_col,
            t_col,
            id_col,
        )


if selected is None:

    REPORT.append(
        "No LPM file containing all three trigger classes was automatically identified."
    )

else:

    (
        ext_path,
        ext,
        ext_q_col,
        ext_t_col,
        ext_id_col,
    ) = selected

    REPORT.append(
        "Selected external file:"
    )

    REPORT.append(
        str(
            ext_path.relative_to(ROOT)
        )
    )

    ext["truth"] = (
        ext[ext_t_col]
        .astype(str)
        .map(LABEL_MAP)
    )

    REPORT.append("")
    REPORT.append(
        "External class counts:"
    )

    REPORT.append(
        ext["truth"]
        .value_counts()
        .to_string()
    )

    # ---------------------------------------------------------
    # Train final classifier on all 1000 primary tasks.
    # Use QUESTION ONLY first because external course-position
    # metadata may be structurally different.
    # ---------------------------------------------------------
    final_q = make_model()

    final_q.fit(
        df["original_question_only"]
        .to_numpy(),
        y,
    )

    ext_questions = (
        ext[ext_q_col]
        .astype(str)
        .to_numpy()
    )

    ext_pred = final_q.predict(
        ext_questions
    )

    ext_y = (
        ext["truth"]
        .to_numpy()
    )

    ext_summary = {
        "N":
            len(ext),

        "accuracy":
            accuracy_score(
                ext_y,
                ext_pred,
            ),

        "balanced_accuracy":
            balanced_accuracy_score(
                ext_y,
                ext_pred,
            ),

        "macro_f1":
            f1_score(
                ext_y,
                ext_pred,
                labels=LABELS,
                average="macro",
                zero_division=0,
            ),

        "false_abstention_rate":
            np.mean(
                ext_pred[
                    ext_y
                    == "answerable"
                ]
                != "answerable"
            ),

        "clinical_miss_rate":
            np.mean(
                ext_pred[
                    ext_y
                    == "clinical_out_of_scope"
                ]
                != "clinical_out_of_scope"
            ),

        "sequence_miss_rate":
            np.mean(
                ext_pred[
                    ext_y
                    == "sequence_sensitive"
                ]
                != "sequence_sensitive"
            ),
    }

    REPORT.append("")
    REPORT.append(
        json.dumps(
            ext_summary,
            indent=2,
        )
    )

    ext_cm = confusion_matrix(
        ext_y,
        ext_pred,
        labels=LABELS,
    )

    ext_cm_df = pd.DataFrame(
        ext_cm,
        index=[
            f"true_{x}"
            for x in LABELS
        ],
        columns=[
            f"pred_{x}"
            for x in LABELS
        ],
    )

    REPORT.append("")
    REPORT.append(
        ext_cm_df.to_string()
    )

    REPORT.append("")
    REPORT.append(
        classification_report(
            ext_y,
            ext_pred,
            labels=LABELS,
            digits=4,
            zero_division=0,
        )
    )

    ext_cm_df.to_csv(
        OUT / "full_lpm_confusion_matrix.csv"
    )

    ext_out = ext.copy()

    ext_out["prediction"] = (
        ext_pred
    )

    ext_out["correct"] = (
        ext_out["truth"]
        == ext_out["prediction"]
    ).astype(int)

    ext_out.to_csv(
        OUT / "full_lpm_trigger_predictions.csv",
        index=False,
    )

    (
        OUT / "full_lpm_selected_file.txt"
    ).write_text(
        str(
            ext_path
        ),
        encoding="utf-8",
    )


# =============================================================================
# FINAL DECISION TABLE
# =============================================================================

section("6. DECISION TABLE")

decision_cols = [
    "experiment",
    "accuracy",
    "balanced_accuracy",
    "macro_f1",
    "false_abstention_rate",
    "clinical_miss_rate",
    "sequence_miss_rate",
]

REPORT.append(
    summary_df[
        decision_cols
    ].to_string(
        index=False
    )
)

REPORT.append("")
REPORT.append(
    "Interpretation:"
)

REPORT.append(
    "E3 reproduces the original deployable-classifier setup."
)

REPORT.append(
    "E6/E7 are the critical stress tests: the classifier is trained on the original benchmark but evaluated on held-out questions after removal of explicit benchmark-family wrappers."
)

REPORT.append(
    "A large E3-to-E7 drop demonstrates template dependence and means the current trigger classifier cannot be presented as solving real-student intent detection."
)

REPORT.append(
    "If sequence recall collapses after cue stripping while clinical detection remains strong, the next method should separate clinical intent detection from course-aware future-concept detection rather than use one three-way text classifier."
)

REPORT.append(
    "The full LPM evaluation is usable only if all three trigger classes are actually represented; a 140-item two-class subset must not be reported as external validation of sequence detection."
)


# =============================================================================
# WRITE REPORT
# =============================================================================

report_text = "\n".join(
    REPORT
)

(
    OUT / "TRIGGER_ROBUSTNESS_AUDIT_REPORT.txt"
).write_text(
    report_text,
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
    "TRIGGER ROBUSTNESS AUDIT COMPLETE"
)
