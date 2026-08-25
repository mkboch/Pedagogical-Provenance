#!/usr/bin/env python3

from pathlib import Path
import json
import math
import os
import re
import sys
import time

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "artifacts/course_aware_trigger_external"
OUT.mkdir(parents=True, exist_ok=True)

PRIMARY_DETAIL = (
    ROOT
    / "artifacts/course_aware_trigger"
    / "course_aware_trigger_detail.csv"
)

LPM_TASKS = (
    ROOT
    / "data/external/lpm/medped_external_candidate"
    / "lpm_external_final_with_sequence.csv"
)

LPM_CORPUS = (
    ROOT
    / "data/external/lpm/medped_external_candidate"
    / "lpm_ordered_slide_corpus.csv"
)

MODEL_NAME = "BAAI/bge-base-en-v1.5"
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

FEATURES = [
    "lex_log_past_docs",
    "lex_log_future_docs",
    "lex_future_fraction",
    "sem_past_max",
    "sem_future_max",
    "sem_delta_future_past",
    "sem_past_top3",
    "sem_future_top3",
    "sem_delta_top3",
    "sem_top10_future_fraction",
    "sem_first_available_rank",
    "position_fraction",
]

REPORT = []


def section(title):
    REPORT.append("")
    REPORT.append("=" * 100)
    REPORT.append(title)
    REPORT.append("=" * 100)


def norm(s):
    s = str(s).lower().strip()
    s = s.replace("–", "-").replace("—", "-")
    s = re.sub(r"[^a-z0-9+\-_/.\s]", " ", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def clean_spaces(s):
    s = re.sub(r"[ \t]+", " ", str(s))
    s = re.sub(r"\s+([,.;:?!])", r"\1", s)
    s = re.sub(r"\n+", " ", s)
    return s.strip(" ,;:-")


def strip_cues(question):
    """
    Remove benchmark wrappers without looking at task_type.
    """
    q = str(question).strip()

    q = re.sub(
        r"^\s*Answer\s+briefly\s+and\s+cite\s+the\s+supporting\s+slide\s+ID\.\s*",
        "",
        q,
        flags=re.I,
    )

    q = re.sub(
        r"^\s*Considering\s+Lecture\s+\d+\s*,?\s*Slide\s+\d+"
        r"(?:\s+about\s+[^,?]+)?\s*,?\s*",
        "",
        q,
        flags=re.I,
    )

    q = re.sub(
        r"^\s*At\s+Lecture\s+\d+\s*,?\s*Slide\s+\d+\s*,?\s*",
        "",
        q,
        flags=re.I,
    )

    q = re.sub(
        r"\s+using\s+only\s+concepts\s+introduced\s+up\s+to\s+that\s+point"
        r"\s+in\s+the\s+course\s*\.?\s*$",
        "",
        q,
        flags=re.I,
    )

    q = re.sub(
        r"\s+using\s+only\s+material\s+available\s+up\s+to\s+that\s+point"
        r"\s*\.?\s*$",
        "",
        q,
        flags=re.I,
    )

    q = re.sub(
        r"\s+based\s+only\s+on\s+what\s+has\s+been\s+introduced\s+so\s+far"
        r"\s*\.?\s*$",
        "",
        q,
        flags=re.I,
    )

    return clean_spaces(q)


GENERIC_STOP = {
    "according", "lecture", "slide", "slides",
    "what", "does", "the", "material", "say",
    "about", "how", "is", "are", "was", "were",
    "explained", "explain", "developed", "across",
    "using", "neighboring", "briefly", "supporting",
    "course", "concepts", "concept", "introduced",
    "point", "only", "from", "in", "at", "of",
    "a", "an", "and", "or", "to", "for", "this",
    "these", "that", "based", "provided",
}


def extract_concept(question):
    q = strip_cues(question)

    patterns = [
        r"what\s+does\s+(?:the\s+)?(?:lecture\s+)?material\s+say\s+about\s+(.+?)[?.]*$",
        r"how\s+is\s+(.+?)\s+explained[?.]*$",
        r"how\s+are\s+(.+?)\s+explained[?.]*$",
        r"explain\s+how\s+(.+?)\s+is\s+developed\s+across.*$",
        r"explain\s+how\s+(.+?)\s+is\s+developed.*$",
        r"explain\s+(.+?)[?.]*$",
    ]

    for pat in patterns:
        m = re.search(pat, q, flags=re.I)

        if m:
            c = m.group(1).strip(" ?.,:;")

            c = re.sub(
                r"\s+using\s+.*$",
                "",
                c,
                flags=re.I,
            )

            c = clean_spaces(c)

            if c:
                return c.lower()

    x = re.sub(
        r"\bLecture\s+\d+\b",
        " ",
        q,
        flags=re.I,
    )

    x = re.sub(
        r"\bSlides?\s+\d+(?:\s*[-–]\s*\d+)?\b",
        " ",
        x,
        flags=re.I,
    )

    tokens = re.findall(
        r"[A-Za-z][A-Za-z0-9+\-]*",
        x.lower(),
    )

    kept = [
        t
        for t in tokens
        if t not in GENERIC_STOP
        and len(t) > 1
    ]

    if kept:
        return " ".join(kept[:8])

    return norm(q)


CLINICAL_PATTERNS = [
    r"\bmedication\s+dosage\b",
    r"\bexact\s+dosage\b",
    r"\bdosage\s+should\b",
    r"\bprescrib(?:e|ed|ing|tion)\b",
    r"\bdiagnos(?:e|ed|is|tic)\b",
    r"\btreatment\s+plan\b",
    r"\btreatment\s+planning\b",
    r"\bpatient[- ]specific\b",
    r"\bspecific\s+patient\b",
    r"\bhospital\s+protocol\b",
    r"\bregulatory\s+guidance\b",
    r"\bclinical\s+decision\b",
    r"\bclinical\s+decision[- ]making\b",
]


def clinical_intent(question):
    q = strip_cues(question)

    return int(
        any(
            re.search(p, q, flags=re.I)
            for p in CLINICAL_PATTERNS
        )
    )


def wilson(successes, n, z=1.959963984540054):
    if n == 0:
        return np.nan, np.nan

    p = successes / n

    denominator = 1.0 + z * z / n

    centre = (
        p
        + z * z / (2.0 * n)
    ) / denominator

    half = (
        z
        * math.sqrt(
            p * (1.0 - p) / n
            + z * z / (4.0 * n * n)
        )
        / denominator
    )

    return (
        max(0.0, centre - half),
        min(1.0, centre + half),
    )


# =============================================================================
# INPUT CHECK
# =============================================================================

for p in [
    PRIMARY_DETAIL,
    LPM_TASKS,
    LPM_CORPUS,
]:
    if not p.exists():
        raise SystemExit(
            f"FAILED: required input missing: {p}"
        )

primary = pd.read_csv(PRIMARY_DETAIL)
ext = pd.read_csv(LPM_TASKS)
corpus = pd.read_csv(LPM_CORPUS)

section("1. INPUT AND SCHEMA AUDIT")

REPORT.append(
    f"Primary feature rows: {len(primary)}"
)

REPORT.append(
    f"External tasks: {len(ext)}"
)

REPORT.append(
    f"LPM corpus rows: {len(corpus)}"
)

REPORT.append("")
REPORT.append(
    "External task columns:"
)

REPORT.append(
    ", ".join(map(str, ext.columns))
)

REPORT.append("")
REPORT.append(
    "LPM corpus columns:"
)

REPORT.append(
    ", ".join(map(str, corpus.columns))
)

print("Primary feature rows:", len(primary))
print("External tasks:", len(ext))
print("LPM corpus:", len(corpus))
print("External columns:", list(ext.columns))
print("Corpus columns:", list(corpus.columns))


# =============================================================================
# PRIMARY TRAINING FEATURES
# =============================================================================

missing_features = [
    c
    for c in FEATURES
    if c not in primary.columns
]

if missing_features:
    raise SystemExit(
        f"FAILED: primary feature file missing {missing_features}"
    )

if "truth" not in primary.columns:
    raise SystemExit(
        "FAILED: primary feature file missing truth column"
    )

train = primary[
    primary["truth"]
    != "clinical_out_of_scope"
].copy()

X_train = (
    train[FEATURES]
    .replace(
        [np.inf, -np.inf],
        np.nan,
    )
    .fillna(0.0)
    .to_numpy()
)

y_train = (
    train["truth"]
    .eq("sequence_sensitive")
    .astype(int)
    .to_numpy()
)

REPORT.append("")
REPORT.append(
    f"Primary sequence-detector training rows: {len(train)}"
)

REPORT.append(
    f"Primary sequence-positive rows: {int(y_train.sum())}"
)

REPORT.append(
    f"Primary answerable rows: {int((y_train == 0).sum())}"
)


# =============================================================================
# EXTERNAL TASK COLUMNS
# =============================================================================

lower_ext = {
    str(c).lower(): c
    for c in ext.columns
}

question_col = None
task_type_col = None
task_id_col = None
target_doc_col = None
target_order_col = None

for c in [
    "question",
    "query",
]:
    if c in lower_ext:
        question_col = lower_ext[c]
        break

for c in [
    "task_type",
    "type",
]:
    if c in lower_ext:
        task_type_col = lower_ext[c]
        break

for c in [
    "task_id",
    "id",
]:
    if c in lower_ext:
        task_id_col = lower_ext[c]
        break

for c in [
    "target_doc_id",
    "target_slide_id",
    "target_id",
    "current_doc_id",
    "current_slide_id",
    "anchor_doc_id",
]:
    if c in lower_ext:
        target_doc_col = lower_ext[c]
        break

for c in [
    "target_order",
    "target_order_index",
    "current_order",
    "current_order_index",
    "sequence_index",
    "position_index",
]:
    if c in lower_ext:
        target_order_col = lower_ext[c]
        break

if question_col is None:
    raise SystemExit(
        "FAILED: could not identify question column"
    )

if task_type_col is None:
    raise SystemExit(
        "FAILED: could not identify task_type column"
    )

if task_id_col is None:
    ext["_task_id"] = [
        f"LPM_{i:04d}"
        for i in range(len(ext))
    ]
    task_id_col = "_task_id"

ext = ext[
    ext[task_type_col]
    .astype(str)
    .isin(LABEL_MAP)
].copy()

ext["truth"] = (
    ext[task_type_col]
    .astype(str)
    .map(LABEL_MAP)
)

ext["question_clean"] = (
    ext[question_col]
    .astype(str)
    .map(strip_cues)
)

ext["concept"] = (
    ext[question_col]
    .astype(str)
    .map(extract_concept)
)

ext["clinical_rule"] = (
    ext[question_col]
    .astype(str)
    .map(clinical_intent)
)

section("2. EXTERNAL TASK AUDIT")

REPORT.append(
    f"Evaluated external tasks: {len(ext)}"
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

REPORT.append("")
REPORT.append(
    "External sequence concepts:"
)

REPORT.append(
    ext[
        ext["truth"]
        .eq("sequence_sensitive")
    ]["concept"]
    .value_counts()
    .to_string()
)


# =============================================================================
# CORPUS COLUMNS
# =============================================================================

lower_corpus = {
    str(c).lower(): c
    for c in corpus.columns
}

doc_col = None
text_col = None
order_col = None

for c in [
    "doc_id",
    "slide_id",
    "document_id",
    "id",
]:
    if c in lower_corpus:
        doc_col = lower_corpus[c]
        break

for c in [
    "text",
    "content",
    "document",
    "transcript",
    "slide_text",
    "combined_text",
]:
    if c in lower_corpus:
        text_col = lower_corpus[c]
        break

for c in [
    "global_order",
    "order_index",
    "sequence_index",
    "course_order",
    "slide_order",
    "position_index",
    "order",
]:
    if c in lower_corpus:
        order_col = lower_corpus[c]
        break

if doc_col is None:
    raise SystemExit(
        "FAILED: could not identify corpus doc ID column"
    )

if text_col is None:
    raise SystemExit(
        "FAILED: could not identify corpus text column"
    )

if order_col is not None:
    corpus["_order"] = pd.to_numeric(
        corpus[order_col],
        errors="coerce",
    )

    if corpus["_order"].isna().any():
        raise SystemExit(
            f"FAILED: order column {order_col} contains nonnumeric values"
        )

    order_source = (
        f"explicit column: {order_col}"
    )

else:
    # This file is explicitly the ordered-slide corpus.
    # If no numeric ordering variable exists, preserve its row order.
    corpus["_order"] = np.arange(
        len(corpus),
        dtype=int,
    )

    order_source = (
        "row order of lpm_ordered_slide_corpus.csv"
    )

corpus["_doc_norm"] = (
    corpus[doc_col]
    .astype(str)
    .map(norm)
)

corpus["_text"] = (
    corpus[text_col]
    .fillna("")
    .astype(str)
)

corpus = (
    corpus.sort_values(
        "_order",
        kind="stable",
    )
    .reset_index(drop=True)
)

doc_to_order = {}

for _, r in corpus.iterrows():
    doc_to_order[
        r["_doc_norm"]
    ] = float(
        r["_order"]
    )

REPORT.append("")
REPORT.append(
    f"Corpus document column: {doc_col}"
)

REPORT.append(
    f"Corpus text column: {text_col}"
)

REPORT.append(
    f"Corpus order source: {order_source}"
)


# =============================================================================
# RESOLVE TARGET COURSE POSITION
# =============================================================================

def map_doc_to_order(value):
    key = norm(value)

    if key in doc_to_order:
        return doc_to_order[key]

    # Relaxed matching against basename / suffix.
    basename = key.split("/")[-1]

    candidates = [
        v
        for k, v in doc_to_order.items()
        if k.endswith("/" + basename)
        or k == basename
    ]

    if len(candidates) == 1:
        return candidates[0]

    return np.nan


if target_order_col is not None:

    ext["_target_order"] = pd.to_numeric(
        ext[target_order_col],
        errors="coerce",
    )

    target_source = (
        f"explicit external column: {target_order_col}"
    )

elif target_doc_col is not None:

    ext["_target_order"] = (
        ext[target_doc_col]
        .map(map_doc_to_order)
    )

    target_source = (
        f"mapped {target_doc_col} -> corpus {doc_col}"
    )

else:

    raise SystemExit(
        "FAILED: external tasks provide neither a target/current order "
        "nor a target/current document identifier. "
        f"Columns were: {list(ext.columns)}"
    )

mapped = int(
    ext["_target_order"]
    .notna()
    .sum()
)

REPORT.append(
    f"Target-position source: {target_source}"
)

REPORT.append(
    f"Target positions mapped: {mapped}/{len(ext)}"
)

if mapped < len(ext):

    bad_cols = [
        task_id_col,
        question_col,
    ]

    if target_doc_col is not None:
        bad_cols.append(
            target_doc_col
        )

    ext[
        ext["_target_order"].isna()
    ][bad_cols].to_csv(
        OUT / "unmapped_external_tasks.csv",
        index=False,
    )

if mapped < int(
    0.95 * len(ext)
):
    raise SystemExit(
        f"FAILED: only {mapped}/{len(ext)} external tasks could be "
        "mapped to course position. Inspect "
        "artifacts/course_aware_trigger_external/unmapped_external_tasks.csv"
    )

# Evaluate only successfully mapped tasks.
ext = ext[
    ext["_target_order"].notna()
].copy()

max_order = float(
    corpus["_order"].max()
)

min_order = float(
    corpus["_order"].min()
)

order_span = max(
    max_order - min_order,
    1.0,
)


# =============================================================================
# LEXICAL COURSE SUPPORT
# =============================================================================

def phrase_pattern(concept):
    c = norm(concept)

    if not c:
        return None

    words = c.split()

    if not words:
        return None

    escaped = []

    for w in words:
        if "-" in w:
            bits = [
                re.escape(x)
                for x in w.split("-")
                if x
            ]

            escaped.append(
                r"[-\s]?".join(bits)
            )
        else:
            escaped.append(
                re.escape(w)
            )

    return re.compile(
        r"\b"
        + r"\s+".join(escaped)
        + r"\b",
        flags=re.I,
    )


corpus_texts = (
    corpus["_text"]
    .astype(str)
    .tolist()
)

corpus_orders = (
    corpus["_order"]
    .astype(float)
    .to_numpy()
)

print(
    "\nComputing external lexical support..."
)

lex_rows = []

for _, r in ext.iterrows():

    pat = phrase_pattern(
        r["concept"]
    )

    target = float(
        r["_target_order"]
    )

    past_docs = 0
    future_docs = 0
    past_mentions = 0
    future_mentions = 0

    if pat is not None:

        for text, order in zip(
            corpus_texts,
            corpus_orders,
        ):

            matches = pat.findall(
                text
            )

            n = len(matches)

            if n == 0:
                continue

            if order <= target:
                past_docs += 1
                past_mentions += n
            else:
                future_docs += 1
                future_mentions += n

    total_docs = (
        past_docs
        + future_docs
    )

    lex_rows.append({
        "lex_past_docs":
            past_docs,

        "lex_future_docs":
            future_docs,

        "lex_past_mentions":
            past_mentions,

        "lex_future_mentions":
            future_mentions,

        "lex_log_past_docs":
            math.log1p(
                past_docs
            ),

        "lex_log_future_docs":
            math.log1p(
                future_docs
            ),

        "lex_future_fraction":
            (
                future_docs
                / total_docs
                if total_docs
                else 0.0
            ),
    })


lex_df = pd.DataFrame(
    lex_rows,
    index=ext.index,
)

for c in lex_df.columns:
    ext[c] = lex_df[c]


# =============================================================================
# BGE SEMANTIC SUPPORT
# =============================================================================

import torch

torch.set_num_threads(
    min(
        8,
        os.cpu_count() or 1,
    )
)

if torch.cuda.is_available():
    raise SystemExit(
        "SAFETY STOP: CUDA must not be visible."
    )

from sentence_transformers import SentenceTransformer

print(
    "\nLoading BGE on CPU:",
    MODEL_NAME
)

embedder = SentenceTransformer(
    MODEL_NAME,
    device="cpu",
)

print(
    "Encoding",
    len(corpus_texts),
    "LPM course documents..."
)

doc_embeddings = embedder.encode(
    corpus_texts,
    batch_size=32,
    normalize_embeddings=True,
    convert_to_numpy=True,
    show_progress_bar=True,
)

concept_inputs = [
    "Represent this sentence for searching relevant passages: "
    + str(c)
    for c in ext["concept"]
]

print(
    "Encoding",
    len(concept_inputs),
    "external concepts..."
)

concept_embeddings = embedder.encode(
    concept_inputs,
    batch_size=32,
    normalize_embeddings=True,
    convert_to_numpy=True,
    show_progress_bar=True,
)


def top_mean(values, k=3):
    values = np.asarray(
        values,
        dtype=float,
    )

    if len(values) == 0:
        return 0.0

    k = min(
        k,
        len(values),
    )

    idx = np.argpartition(
        values,
        len(values) - k,
    )[-k:]

    return float(
        np.mean(
            values[idx]
        )
    )


print(
    "Computing external semantic availability features..."
)

sem_rows = []

for i, (_, r) in enumerate(
    ext.iterrows()
):

    target = float(
        r["_target_order"]
    )

    scores = (
        doc_embeddings
        @ concept_embeddings[i]
    )

    past_mask = (
        corpus_orders
        <= target
    )

    future_mask = (
        corpus_orders
        > target
    )

    past_scores = scores[
        past_mask
    ]

    future_scores = scores[
        future_mask
    ]

    past_max = (
        float(
            past_scores.max()
        )
        if len(past_scores)
        else 0.0
    )

    future_max = (
        float(
            future_scores.max()
        )
        if len(future_scores)
        else 0.0
    )

    past_top3 = top_mean(
        past_scores,
        3,
    )

    future_top3 = top_mean(
        future_scores,
        3,
    )

    ranking = np.argsort(
        -scores,
        kind="stable",
    )

    top10 = ranking[:10]

    top10_future_fraction = float(
        np.mean(
            corpus_orders[top10]
            > target
        )
    )

    first_available_rank = 0

    for rank, idx in enumerate(
        ranking,
        start=1,
    ):

        if (
            corpus_orders[idx]
            <= target
        ):
            first_available_rank = rank
            break

    sem_rows.append({
        "sem_past_max":
            past_max,

        "sem_future_max":
            future_max,

        "sem_delta_future_past":
            future_max
            - past_max,

        "sem_past_top3":
            past_top3,

        "sem_future_top3":
            future_top3,

        "sem_delta_top3":
            future_top3
            - past_top3,

        "sem_top10_future_fraction":
            top10_future_fraction,

        "sem_first_available_rank":
            first_available_rank,

        "position_fraction":
            (
                target
                - min_order
            )
            / order_span,
    })


sem_df = pd.DataFrame(
    sem_rows,
    index=ext.index,
)

for c in sem_df.columns:
    ext[c] = sem_df[c]


# =============================================================================
# TRAIN PRIMARY SEQUENCE MODEL
#
# No external labels are used for training, thresholding, calibration,
# model selection, or feature selection.
# =============================================================================

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

sequence_model = Pipeline([
    (
        "scale",
        StandardScaler(),
    ),
    (
        "logreg",
        LogisticRegression(
            C=1.0,
            class_weight="balanced",
            max_iter=3000,
            random_state=SEED,
        ),
    ),
])

sequence_model.fit(
    X_train,
    y_train,
)


# =============================================================================
# EXTERNAL PREDICTION
# =============================================================================

X_ext = (
    ext[FEATURES]
    .replace(
        [np.inf, -np.inf],
        np.nan,
    )
    .fillna(0.0)
    .to_numpy()
)

sequence_prob = (
    sequence_model
    .predict_proba(
        X_ext
    )[:, 1]
)

ext[
    "sequence_probability"
] = sequence_prob

pred = []

for i, (_, r) in enumerate(
    ext.iterrows()
):

    if int(
        r["clinical_rule"]
    ) == 1:

        pred.append(
            "clinical_out_of_scope"
        )

    elif sequence_prob[i] >= 0.5:

        pred.append(
            "sequence_sensitive"
        )

    else:

        pred.append(
            "answerable"
        )

ext["prediction"] = pred

ext["correct"] = (
    ext["truth"]
    ==
    ext["prediction"]
).astype(int)


# =============================================================================
# METRICS
# =============================================================================

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

truth = (
    ext["truth"]
    .to_numpy()
)

prediction = (
    ext["prediction"]
    .to_numpy()
)

answer_mask = (
    truth
    == "answerable"
)

clinical_mask = (
    truth
    == "clinical_out_of_scope"
)

sequence_mask = (
    truth
    == "sequence_sensitive"
)

answer_correct = int(
    np.sum(
        prediction[answer_mask]
        == "answerable"
    )
)

clinical_correct = int(
    np.sum(
        prediction[clinical_mask]
        == "clinical_out_of_scope"
    )
)

sequence_correct = int(
    np.sum(
        prediction[sequence_mask]
        == "sequence_sensitive"
    )
)

answer_n = int(
    np.sum(
        answer_mask
    )
)

clinical_n = int(
    np.sum(
        clinical_mask
    )
)

sequence_n = int(
    np.sum(
        sequence_mask
    )
)

answer_ci = wilson(
    answer_correct,
    answer_n,
)

clinical_ci = wilson(
    clinical_correct,
    clinical_n,
)

sequence_ci = wilson(
    sequence_correct,
    sequence_n,
)

summary = {
    "N":
        len(ext),

    "accuracy":
        accuracy_score(
            truth,
            prediction,
        ),

    "balanced_accuracy":
        balanced_accuracy_score(
            truth,
            prediction,
        ),

    "macro_f1":
        f1_score(
            truth,
            prediction,
            labels=LABELS,
            average="macro",
            zero_division=0,
        ),

    "answerable_recall":
        (
            answer_correct
            / answer_n
            if answer_n
            else np.nan
        ),

    "false_abstention_rate":
        (
            1.0
            - answer_correct
            / answer_n
            if answer_n
            else np.nan
        ),

    "clinical_recall":
        (
            clinical_correct
            / clinical_n
            if clinical_n
            else np.nan
        ),

    "clinical_miss_rate":
        (
            1.0
            - clinical_correct
            / clinical_n
            if clinical_n
            else np.nan
        ),

    "sequence_recall":
        (
            sequence_correct
            / sequence_n
            if sequence_n
            else np.nan
        ),

    "sequence_miss_rate":
        (
            1.0
            - sequence_correct
            / sequence_n
            if sequence_n
            else np.nan
        ),

    "answerable_recall_ci_low":
        answer_ci[0],

    "answerable_recall_ci_high":
        answer_ci[1],

    "clinical_recall_ci_low":
        clinical_ci[0],

    "clinical_recall_ci_high":
        clinical_ci[1],

    "sequence_recall_ci_low":
        sequence_ci[0],

    "sequence_recall_ci_high":
        sequence_ci[1],
}

summary_df = pd.DataFrame([
    summary
])

summary_df.to_csv(
    OUT
    / "external_course_aware_trigger_summary.csv",
    index=False,
)


# =============================================================================
# CONFUSION MATRIX
# =============================================================================

cm = confusion_matrix(
    truth,
    prediction,
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
    OUT
    / "external_course_aware_trigger_confusion_matrix.csv"
)


# =============================================================================
# SEQUENCE CONCEPT RESULTS
# =============================================================================

seq_detail = ext[
    ext["truth"]
    .eq(
        "sequence_sensitive"
    )
].copy()

seq_by_concept = (
    seq_detail.groupby(
        "concept"
    )
    .agg(
        N=(
            task_id_col,
            "count",
        ),

        Correct=(
            "correct",
            "sum",
        ),

        MeanSequenceProbability=(
            "sequence_probability",
            "mean",
        ),

        MeanPastDocs=(
            "lex_past_docs",
            "mean",
        ),

        MeanFutureDocs=(
            "lex_future_docs",
            "mean",
        ),

        MeanPastSemantic=(
            "sem_past_max",
            "mean",
        ),

        MeanFutureSemantic=(
            "sem_future_max",
            "mean",
        ),
    )
    .reset_index()
)

seq_by_concept[
    "Recall"
] = (
    seq_by_concept[
        "Correct"
    ]
    /
    seq_by_concept[
        "N"
    ]
)

seq_by_concept.to_csv(
    OUT
    / "external_sequence_results_by_concept.csv",
    index=False,
)


# =============================================================================
# FEATURE-SHIFT AUDIT
# =============================================================================

primary_nonclinical = primary[
    primary["truth"]
    != "clinical_out_of_scope"
].copy()

external_nonclinical = ext[
    ext["truth"]
    != "clinical_out_of_scope"
].copy()

feature_shift_rows = []

for feature in FEATURES:

    pvals = pd.to_numeric(
        primary_nonclinical[
            feature
        ],
        errors="coerce",
    ).dropna()

    evals = pd.to_numeric(
        external_nonclinical[
            feature
        ],
        errors="coerce",
    ).dropna()

    feature_shift_rows.append({
        "feature":
            feature,

        "primary_mean":
            pvals.mean(),

        "primary_std":
            pvals.std(),

        "external_mean":
            evals.mean(),

        "external_std":
            evals.std(),

        "standardized_mean_shift":
            (
                (
                    evals.mean()
                    - pvals.mean()
                )
                /
                max(
                    float(
                        pvals.std()
                    ),
                    1e-12,
                )
            ),
    })


feature_shift = pd.DataFrame(
    feature_shift_rows
)

feature_shift.to_csv(
    OUT
    / "primary_to_external_feature_shift.csv",
    index=False,
)


# =============================================================================
# SAVE FULL DETAIL
# =============================================================================

ext.to_csv(
    OUT
    / "external_course_aware_trigger_detail.csv",
    index=False,
)

errors = ext[
    ext["correct"]
    .eq(0)
].copy()

errors.to_csv(
    OUT
    / "external_course_aware_trigger_errors.csv",
    index=False,
)


# =============================================================================
# REPORT
# =============================================================================

section("3. PRIMARY-TO-LPM COURSE-AWARE TRANSFER")

REPORT.append(
    "Training source: primary 1,000-task benchmark; sequence model trained only on primary answerable/sequence tasks."
)

REPORT.append(
    "External source: 155-item manually reviewed LPM set."
)

REPORT.append(
    "No LPM task labels were used for training, feature selection, threshold selection, or calibration."
)

REPORT.append(
    "Sequence decision uses evidence-availability features rather than raw question tokens or requested concept identity."
)

REPORT.append("")
REPORT.append(
    json.dumps(
        summary,
        indent=2,
    )
)

REPORT.append("")
REPORT.append(
    "Confusion matrix:"
)

REPORT.append(
    cm_df.to_string()
)

REPORT.append("")
REPORT.append(
    classification_report(
        truth,
        prediction,
        labels=LABELS,
        digits=4,
        zero_division=0,
    )
)


section("4. EXTERNAL SEQUENCE CONCEPT RESULTS")

REPORT.append(
    seq_by_concept.to_string(
        index=False,
    )
)


section("5. FEATURE-SHIFT AUDIT")

REPORT.append(
    feature_shift.to_string(
        index=False,
    )
)


section("6. ERROR ANALYSIS")

REPORT.append(
    f"Errors: {len(errors)} / {len(ext)}"
)

if len(errors):

    REPORT.append("")
REPORT.append(
    "Error transitions:"
)

if len(errors):
    REPORT.append(
        errors.groupby(
            [
                "truth",
                "prediction",
            ]
        )
        .size()
        .to_string()
    )

    REPORT.append("")
    REPORT.append(
        "All errors:"
    )

    for _, r in errors.iterrows():

        REPORT.append(
            f"{r[task_id_col]} | "
            f"true={r['truth']} | "
            f"pred={r['prediction']} | "
            f"concept={r['concept']} | "
            f"p_seq={r['sequence_probability']:.4f} | "
            f"past_docs={r['lex_past_docs']} | "
            f"future_docs={r['lex_future_docs']} | "
            f"sem_past={r['sem_past_max']:.4f} | "
            f"sem_future={r['sem_future_max']:.4f} | "
            f"Q={r['question_clean']}"
        )


section("7. DECISION")

REPORT.append(
    "If sequence recall transfers to the 15 LPM sequence concepts while false abstention remains acceptable, "
    "the course-aware trigger can be propagated through the fixed verifier as the deployable condition."
)

REPORT.append(
    "If primary performance is high but LPM sequence recall collapses, the verifier should remain an oracle-control upper bound; "
    "the paper should not claim deployable sequence gating."
)

REPORT.append(
    "Because the external sequence sample contains only 15 items, the Wilson confidence interval must be reported alongside its point estimate."
)


report_text = "\n".join(
    REPORT
)

(
    OUT
    / "COURSE_AWARE_TRIGGER_EXTERNAL_REPORT.txt"
).write_text(
    report_text,
    encoding="utf-8",
)

metadata = {
    "primary_feature_file":
        str(PRIMARY_DETAIL),

    "external_tasks":
        str(LPM_TASKS),

    "external_corpus":
        str(LPM_CORPUS),

    "embedding_model":
        MODEL_NAME,

    "features":
        FEATURES,

    "external_labels_used_for_training":
        False,

    "raw_question_tokens_used_by_sequence_model":
        False,

    "concept_identity_used_by_sequence_model":
        False,

    "threshold":
        0.5,

    "target_position_source":
        target_source,

    "corpus_order_source":
        order_source,

    "mapped_external_tasks":
        len(ext),

    "cuda_visible_devices":
        os.environ.get(
            "CUDA_VISIBLE_DEVICES"
        ),
}

(
    OUT
    / "course_aware_trigger_external_metadata.json"
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
    "PRIMARY-TO-LPM COURSE-AWARE TRIGGER TRANSFER COMPLETE"
)
