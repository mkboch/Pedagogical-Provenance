#!/usr/bin/env python3

from pathlib import Path
from collections import Counter
import ast
import json
import math
import os
import re
import time

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "artifacts/course_aware_trigger"
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


def normalize_text(s):
    s = str(s).lower()
    s = s.replace("–", "-")
    s = s.replace("—", "-")
    s = re.sub(r"[^a-z0-9+\-\s]", " ", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip()


# =============================================================================
# CUE STRIPPING
# =============================================================================

def clean_spaces(s):
    s = re.sub(r"[ \t]+", " ", str(s))
    s = re.sub(r"\s+([,.;:?!])", r"\1", s)
    s = re.sub(r"\n+", " ", s)
    return s.strip(" ,;:-")


def strip_cues(question):
    """
    Generic removal of benchmark wrappers.

    This function does NOT receive task_type.
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

    return clean_spaces(q)


# =============================================================================
# CONCEPT EXTRACTION
#
# Important:
# - no task_type
# - no future_keyword
# - no anchor_term
# - no gold evidence
#
# It extracts the semantic object of the student's request.
# =============================================================================

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
    q = strip_cues(question).strip()

    # Most generic semantic forms first.
    patterns = [
        r"what\s+does\s+(?:the\s+)?(?:lecture\s+)?material\s+say\s+about\s+(.+?)[?.]*$",
        r"how\s+is\s+(.+?)\s+explained[?.]*$",
        r"how\s+are\s+(.+?)\s+explained[?.]*$",
        r"explain\s+how\s+(.+?)\s+is\s+developed\s+across.*$",
        r"explain\s+how\s+(.+?)\s+is\s+developed.*$",
        r"explain\s+(.+?)[?.]*$",
    ]

    for pat in patterns:
        m = re.search(
            pat,
            q,
            flags=re.I,
        )

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

    # Fallback: remove explicit lecture/slide material and generic words.
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
        return " ".join(
            kept[:5]
        )

    return normalize_text(q)


# =============================================================================
# CLINICAL INTENT DETECTOR
#
# Separate from sequence detection.
#
# These are actual clinical-action semantics, not benchmark-position cues.
# =============================================================================

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
            re.search(
                p,
                q,
                flags=re.I,
            )
            for p in CLINICAL_PATTERNS
        )
    )


# =============================================================================
# LOAD PRIMARY TASKS
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
        ROOT.glob(
            "**/full1000_all_models_metrics_detail.csv"
        )
    )

    if not matches:
        raise SystemExit(
            "FAILED: primary detail CSV not found"
        )

    DETAIL = matches[0]

raw = pd.read_csv(
    DETAIL
)

required = [
    "task_id",
    "task_type",
    "question",
    "target_doc_id",
]

missing = [
    c for c in required
    if c not in raw.columns
]

if missing:
    raise SystemExit(
        f"FAILED: missing columns {missing}"
    )

tasks = (
    raw[required]
    .drop_duplicates("task_id")
    .copy()
)

tasks = tasks[
    tasks["task_type"].isin(
        LABEL_MAP
    )
].copy()

tasks["truth"] = (
    tasks["task_type"]
    .map(LABEL_MAP)
)

pos = (
    tasks["target_doc_id"]
    .map(parse_position)
)

tasks["lecture"] = pos.map(
    lambda x: x[0] if x else np.nan
)

tasks["slide"] = pos.map(
    lambda x: x[1] if x else np.nan
)

tasks = tasks.dropna(
    subset=[
        "lecture",
        "slide",
    ]
).copy()

tasks["lecture"] = (
    tasks["lecture"]
    .astype(int)
)

tasks["slide"] = (
    tasks["slide"]
    .astype(int)
)

tasks["question_stripped"] = (
    tasks["question"]
    .map(strip_cues)
)

tasks["concept"] = (
    tasks["question"]
    .map(extract_concept)
)

tasks["clinical_rule"] = (
    tasks["question"]
    .map(clinical_intent)
)

if len(tasks) != 1000:
    raise SystemExit(
        f"FAILED: expected 1000 tasks, got {len(tasks)}"
    )


# =============================================================================
# LOAD PRIMARY COURSE CORPUS
# =============================================================================

CORPUS = (
    ROOT
    / "artifacts/retrieval_inputs"
    / "discovered_slide_corpus.csv"
)

if not CORPUS.exists():
    matches = sorted(
        ROOT.glob(
            "**/discovered_slide_corpus.csv"
        )
    )

    if not matches:
        raise SystemExit(
            "FAILED: primary corpus not found"
        )

    CORPUS = matches[0]

cdf = pd.read_csv(
    CORPUS
)

docs_by_id = {}

for _, r in cdf.iterrows():

    did = canonical_doc_id(
        r["doc_id"]
    )

    p = parse_position(
        did
    )

    if p is None:
        continue

    text = str(
        r["text"]
    )

    item = {
        "doc_id": did,
        "lecture": p[0],
        "slide": p[1],
        "pos": p,
        "text": text,
        "norm_text": normalize_text(text),
    }

    if (
        did not in docs_by_id
        or len(text)
        > len(
            docs_by_id[did]["text"]
        )
    ):
        docs_by_id[did] = item

docs = sorted(
    docs_by_id.values(),
    key=lambda x: (
        x["lecture"],
        x["slide"],
        x["doc_id"],
    ),
)

texts = [
    d["text"]
    for d in docs
]

print(
    "Primary tasks:",
    len(tasks)
)

print(
    "Primary corpus:",
    len(docs)
)


# =============================================================================
# CONCEPT EXTRACTION AUDIT
# =============================================================================

section("1. CONCEPT EXTRACTION AUDIT")

concept_summary = (
    tasks.groupby(
        [
            "task_type",
            "concept",
        ]
    )
    .size()
    .reset_index(
        name="N"
    )
    .sort_values(
        [
            "task_type",
            "N",
        ],
        ascending=[
            True,
            False,
        ],
    )
)

concept_summary.to_csv(
    OUT / "concept_extraction_counts.csv",
    index=False,
)

REPORT.append(
    f"Tasks: {len(tasks)}"
)

REPORT.append(
    f"Corpus slides: {len(docs)}"
)

REPORT.append("")
REPORT.append(
    "Top extracted concepts by task family:"
)

for tt in LABEL_MAP:

    REPORT.append("")
    REPORT.append(
        f"[{tt}]"
    )

    x = concept_summary[
        concept_summary[
            "task_type"
        ].eq(tt)
    ].head(20)

    REPORT.append(
        x.to_string(
            index=False
        )
    )


# =============================================================================
# CLINICAL DETECTOR AUDIT
# =============================================================================

section("2. CLINICAL INTENT DETECTOR")

clinical_truth = (
    tasks["truth"]
    .eq(
        "clinical_out_of_scope"
    )
    .astype(int)
    .to_numpy()
)

clinical_pred = (
    tasks["clinical_rule"]
    .to_numpy()
)

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

clinical_stats = {
    "accuracy":
        accuracy_score(
            clinical_truth,
            clinical_pred,
        ),

    "precision":
        precision_score(
            clinical_truth,
            clinical_pred,
            zero_division=0,
        ),

    "recall":
        recall_score(
            clinical_truth,
            clinical_pred,
            zero_division=0,
        ),

    "f1":
        f1_score(
            clinical_truth,
            clinical_pred,
            zero_division=0,
        ),

    "false_positive_rate_on_nonclinical":
        float(
            np.mean(
                clinical_pred[
                    clinical_truth == 0
                ]
                == 1
            )
        ),
}

REPORT.append(
    json.dumps(
        clinical_stats,
        indent=2,
    )
)


# =============================================================================
# LEXICAL COURSE-SUPPORT FEATURES
# =============================================================================

def phrase_pattern(concept):
    c = normalize_text(
        concept
    )

    if not c:
        return None

    words = c.split()

    if not words:
        return None

    # Flexible hyphen/space matching for terms such as k-space.
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
        + r"\s+".join(
            escaped
        )
        + r"\b",
        flags=re.I,
    )


def lexical_support(
    concept,
    target_pos,
):
    pat = phrase_pattern(
        concept
    )

    if pat is None:
        return {
            "lex_past_docs": 0,
            "lex_future_docs": 0,
            "lex_past_mentions": 0,
            "lex_future_mentions": 0,
            "lex_first_lecture": -1,
            "lex_first_slide": -1,
        }

    past_docs = 0
    future_docs = 0

    past_mentions = 0
    future_mentions = 0

    first = None

    for d in docs:

        matches = pat.findall(
            d["text"]
        )

        count = len(
            matches
        )

        if count <= 0:
            continue

        if first is None:
            first = d["pos"]

        if (
            d["pos"]
            <= target_pos
        ):
            past_docs += 1
            past_mentions += count
        else:
            future_docs += 1
            future_mentions += count

    return {
        "lex_past_docs":
            past_docs,

        "lex_future_docs":
            future_docs,

        "lex_past_mentions":
            past_mentions,

        "lex_future_mentions":
            future_mentions,

        "lex_first_lecture":
            first[0]
            if first
            else -1,

        "lex_first_slide":
            first[1]
            if first
            else -1,
    }


print(
    "\nComputing lexical course-support features..."
)

lex_features = []

for i, r in tasks.iterrows():

    target_pos = (
        int(r["lecture"]),
        int(r["slide"]),
    )

    f = lexical_support(
        r["concept"],
        target_pos,
    )

    lex_features.append(
        f
    )

lex_df = pd.DataFrame(
    lex_features,
    index=tasks.index,
)

for c in lex_df.columns:
    tasks[c] = lex_df[c]


# =============================================================================
# BGE SEMANTIC COURSE-SUPPORT FEATURES
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

MODEL_NAME = (
    "BAAI/bge-base-en-v1.5"
)

print(
    "Loading BGE on CPU:",
    MODEL_NAME
)

model = SentenceTransformer(
    MODEL_NAME,
    device="cpu",
)

doc_embeddings = model.encode(
    texts,
    batch_size=32,
    normalize_embeddings=True,
    convert_to_numpy=True,
    show_progress_bar=True,
)

concept_inputs = [
    "Represent this sentence for searching relevant passages: "
    + str(c)
    for c in tasks["concept"]
]

concept_embeddings = model.encode(
    concept_inputs,
    batch_size=32,
    normalize_embeddings=True,
    convert_to_numpy=True,
    show_progress_bar=True,
)


def top_mean(x, k=3):
    if len(x) == 0:
        return 0.0

    k = min(
        k,
        len(x),
    )

    vals = np.partition(
        x,
        len(x) - k,
    )[
        -k:
    ]

    return float(
        np.mean(vals)
    )


print(
    "Computing semantic availability features..."
)

semantic_rows = []

for ti, (_, r) in enumerate(
    tasks.iterrows()
):

    target_pos = (
        int(r["lecture"]),
        int(r["slide"]),
    )

    scores = (
        doc_embeddings
        @ concept_embeddings[ti]
    )

    past_idx = np.array(
        [
            j
            for j, d in enumerate(docs)
            if d["pos"]
            <= target_pos
        ],
        dtype=int,
    )

    future_idx = np.array(
        [
            j
            for j, d in enumerate(docs)
            if d["pos"]
            > target_pos
        ],
        dtype=int,
    )

    past_scores = (
        scores[past_idx]
        if len(past_idx)
        else np.array([])
    )

    future_scores = (
        scores[future_idx]
        if len(future_idx)
        else np.array([])
    )

    past_max = (
        float(
            np.max(
                past_scores
            )
        )
        if len(past_scores)
        else 0.0
    )

    future_max = (
        float(
            np.max(
                future_scores
            )
        )
        if len(future_scores)
        else 0.0
    )

    order = np.argsort(
        -scores,
        kind="stable",
    )

    top10 = order[:10]

    top10_future_fraction = float(
        np.mean([
            docs[j]["pos"]
            > target_pos
            for j in top10
        ])
    )

    first_available_rank = 0

    for rank, j in enumerate(
        order,
        start=1,
    ):
        if (
            docs[j]["pos"]
            <= target_pos
        ):
            first_available_rank = rank
            break

    semantic_rows.append({
        "sem_past_max":
            past_max,

        "sem_future_max":
            future_max,

        "sem_delta_future_past":
            future_max
            - past_max,

        "sem_past_top3":
            top_mean(
                past_scores,
                3,
            ),

        "sem_future_top3":
            top_mean(
                future_scores,
                3,
            ),

        "sem_delta_top3":
            top_mean(
                future_scores,
                3,
            )
            -
            top_mean(
                past_scores,
                3,
            ),

        "sem_top10_future_fraction":
            top10_future_fraction,

        "sem_first_available_rank":
            first_available_rank,
    })

sem_df = pd.DataFrame(
    semantic_rows,
    index=tasks.index,
)

for c in sem_df.columns:
    tasks[c] = sem_df[c]


# =============================================================================
# DERIVED SUPPORT FEATURES
# =============================================================================

tasks[
    "lex_total_docs"
] = (
    tasks["lex_past_docs"]
    + tasks["lex_future_docs"]
)

tasks[
    "lex_future_fraction"
] = np.where(
    tasks["lex_total_docs"]
    > 0,
    tasks["lex_future_docs"]
    /
    tasks["lex_total_docs"],
    0.0,
)

tasks[
    "lex_log_past_docs"
] = np.log1p(
    tasks["lex_past_docs"]
)

tasks[
    "lex_log_future_docs"
] = np.log1p(
    tasks["lex_future_docs"]
)

tasks[
    "position_fraction"
] = (
    tasks["lecture"]
    / max(
        tasks["lecture"].max(),
        1,
    )
)


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


# =============================================================================
# COURSE-SUPPORT SEQUENCE CLASSIFIER
#
# Critical design:
# raw words/concept identity are NOT input features.
#
# It sees only:
# - how much support exists before current position
# - how much support exists after current position
# - relative semantic support
# - current course progress
#
# Therefore it cannot memorize "ultrasound" or "k-space" as class labels.
# =============================================================================

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedGroupKFold

nonclinical = tasks[
    tasks["truth"]
    != "clinical_out_of_scope"
].copy()

X = (
    nonclinical[
        FEATURES
    ]
    .replace(
        [
            np.inf,
            -np.inf,
        ],
        np.nan,
    )
    .fillna(0.0)
    .to_numpy()
)

y = (
    nonclinical["truth"]
    .eq(
        "sequence_sensitive"
    )
    .astype(int)
    .to_numpy()
)

groups = (
    nonclinical["lecture"]
    .to_numpy()
)

sgkf = StratifiedGroupKFold(
    n_splits=5,
    shuffle=True,
    random_state=SEED,
)

seq_oof = np.zeros(
    len(nonclinical),
    dtype=int,
)

seq_prob = np.zeros(
    len(nonclinical),
    dtype=float,
)

fold_id = np.zeros(
    len(nonclinical),
    dtype=int,
)

print(
    "\nRunning lecture-grouped course-support CV..."
)

for fold, (
    tr,
    te,
) in enumerate(
    sgkf.split(
        X,
        y,
        groups,
    ),
    start=1,
):

    clf = Pipeline([
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

    clf.fit(
        X[tr],
        y[tr],
    )

    p = clf.predict_proba(
        X[te]
    )[:, 1]

    seq_prob[te] = p
    seq_oof[te] = (
        p >= 0.5
    ).astype(int)

    fold_id[te] = fold


nonclinical[
    "sequence_support_probability"
] = seq_prob

nonclinical[
    "sequence_support_prediction"
] = seq_oof

nonclinical[
    "cv_fold"
] = fold_id


# =============================================================================
# FINAL HYBRID OOF PREDICTIONS
#
# clinical intent first
# then course-support sequence detector
# otherwise answerable
# =============================================================================

seq_lookup = {
    str(r["task_id"]):
        int(
            r[
                "sequence_support_prediction"
            ]
        )
    for _, r in nonclinical.iterrows()
}

seq_prob_lookup = {
    str(r["task_id"]):
        float(
            r[
                "sequence_support_probability"
            ]
        )
    for _, r in nonclinical.iterrows()
}

hybrid_pred = []
hybrid_prob = []

for _, r in tasks.iterrows():

    tid = str(
        r["task_id"]
    )

    if int(
        r["clinical_rule"]
    ) == 1:

        hybrid_pred.append(
            "clinical_out_of_scope"
        )

        hybrid_prob.append(
            np.nan
        )

    else:

        seq = seq_lookup[
            tid
        ]

        prob = seq_prob_lookup[
            tid
        ]

        hybrid_pred.append(
            "sequence_sensitive"
            if seq
            else "answerable"
        )

        hybrid_prob.append(
            prob
        )


tasks[
    "hybrid_prediction"
] = hybrid_pred

tasks[
    "sequence_probability"
] = hybrid_prob

tasks[
    "correct"
] = (
    tasks["truth"]
    ==
    tasks["hybrid_prediction"]
).astype(int)


# =============================================================================
# LABEL-FREE RULE ABLATIONS
# =============================================================================

rules = {
    # No prior exact lexical support but later support exists.
    "R1_zero_past_exact":
        (
            (
                tasks["lex_past_docs"]
                == 0
            )
            &
            (
                tasks["lex_future_docs"]
                > 0
            )
        ),

    # Fewer than two prior supporting slides but repeated future support.
    "R2_low_past_exact":
        (
            (
                tasks["lex_past_docs"]
                < 2
            )
            &
            (
                tasks["lex_future_docs"]
                >= 2
            )
        ),

    # Future semantic match is substantially stronger.
    "R3_semantic_margin_005":
        (
            tasks[
                "sem_delta_future_past"
            ]
            >= 0.05
        ),

    "R4_semantic_margin_010":
        (
            tasks[
                "sem_delta_future_past"
            ]
            >= 0.10
        ),

    # Conservative combined rule.
    "R5_lexical_semantic":
        (
            (
                tasks["lex_past_docs"]
                < 2
            )
            &
            (
                tasks["lex_future_docs"]
                >= 2
            )
            &
            (
                tasks[
                    "sem_delta_future_past"
                ]
                > 0
            )
        ),
}


# =============================================================================
# METRICS
# =============================================================================

def multiclass_stats(
    truth,
    pred,
    name,
):
    truth = np.asarray(
        truth
    )

    pred = np.asarray(
        pred
    )

    answer = (
        truth
        == "answerable"
    )

    clinical = (
        truth
        == "clinical_out_of_scope"
    )

    sequence = (
        truth
        == "sequence_sensitive"
    )

    return {
        "method":
            name,

        "N":
            len(truth),

        "accuracy":
            accuracy_score(
                truth,
                pred,
            ),

        "balanced_accuracy":
            balanced_accuracy_score(
                truth,
                pred,
            ),

        "macro_f1":
            f1_score(
                truth,
                pred,
                labels=LABELS,
                average="macro",
                zero_division=0,
            ),

        "false_abstention_answerable":
            float(
                np.mean(
                    pred[answer]
                    != "answerable"
                )
            ),

        "clinical_miss":
            float(
                np.mean(
                    pred[clinical]
                    != "clinical_out_of_scope"
                )
            ),

        "sequence_miss":
            float(
                np.mean(
                    pred[sequence]
                    != "sequence_sensitive"
                )
            ),

        "sequence_recall":
            float(
                np.mean(
                    pred[sequence]
                    == "sequence_sensitive"
                )
            ),
    }


summary_rows = []

summary_rows.append(
    multiclass_stats(
        tasks["truth"],
        tasks[
            "hybrid_prediction"
        ],
        "learned_course_support_hybrid",
    )
)

for rule_name, mask in rules.items():

    pred = []

    for i, (_, r) in enumerate(
        tasks.iterrows()
    ):

        if (
            int(
                r["clinical_rule"]
            )
            == 1
        ):

            pred.append(
                "clinical_out_of_scope"
            )

        elif bool(
            mask.iloc[i]
        ):

            pred.append(
                "sequence_sensitive"
            )

        else:

            pred.append(
                "answerable"
            )

    tasks[
        f"prediction_{rule_name}"
    ] = pred

    summary_rows.append(
        multiclass_stats(
            tasks["truth"],
            pred,
            rule_name,
        )
    )


summary_df = pd.DataFrame(
    summary_rows
)

summary_df.to_csv(
    OUT
    / "course_aware_trigger_summary.csv",
    index=False,
)


# =============================================================================
# CONFUSION MATRIX
# =============================================================================

cm = confusion_matrix(
    tasks["truth"],
    tasks[
        "hybrid_prediction"
    ],
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
    / "course_aware_hybrid_confusion_matrix.csv"
)


# =============================================================================
# TASK-LEVEL OUTPUT
# =============================================================================

tasks.to_csv(
    OUT
    / "course_aware_trigger_detail.csv",
    index=False,
)

errors = tasks[
    tasks["correct"]
    .eq(0)
].copy()

errors.to_csv(
    OUT
    / "course_aware_trigger_errors.csv",
    index=False,
)


# =============================================================================
# FEATURE SUMMARY BY TRUE CLASS
# =============================================================================

feature_summary = (
    tasks.groupby(
        "truth"
    )[
        FEATURES
        + [
            "lex_past_docs",
            "lex_future_docs",
        ]
    ]
    .mean()
    .reset_index()
)

feature_summary.to_csv(
    OUT
    / "course_support_feature_summary.csv",
    index=False,
)


# =============================================================================
# SEQUENCE-CONCEPT ANALYSIS
# =============================================================================

sequence_concepts = (
    tasks[
        tasks["truth"]
        .eq(
            "sequence_sensitive"
        )
    ]
    .groupby(
        "concept"
    )
    .agg(
        N=("task_id", "count"),
        MeanPastDocs=(
            "lex_past_docs",
            "mean"
        ),
        MeanFutureDocs=(
            "lex_future_docs",
            "mean"
        ),
        MeanPastSemantic=(
            "sem_past_max",
            "mean"
        ),
        MeanFutureSemantic=(
            "sem_future_max",
            "mean"
        ),
        HybridRecall=(
            "correct",
            "mean"
        ),
    )
    .reset_index()
)

sequence_concepts.to_csv(
    OUT
    / "primary_sequence_concept_analysis.csv",
    index=False,
)


# =============================================================================
# LPM SEQUENCE CONCEPT DIVERSITY AUDIT
#
# No prediction yet.
# We first establish whether the external sequence set introduces
# different requested concepts.
# =============================================================================

LPM = (
    ROOT
    / "data/external/lpm/medped_external_candidate"
    / "lpm_external_final_with_sequence.csv"
)

lpm_concept_df = pd.DataFrame()

if LPM.exists():

    ext = pd.read_csv(
        LPM
    )

    if (
        "task_type"
        in ext.columns
        and
        "question"
        in ext.columns
    ):

        ext[
            "question_stripped"
        ] = (
            ext["question"]
            .map(strip_cues)
        )

        ext[
            "concept"
        ] = (
            ext["question"]
            .map(extract_concept)
        )

        seq_ext = ext[
            ext["task_type"]
            .eq(
                "lecture_sequence_violation_probe"
            )
        ].copy()

        lpm_concept_df = (
            seq_ext.groupby(
                "concept"
            )
            .size()
            .reset_index(
                name="N"
            )
            .sort_values(
                "N",
                ascending=False,
            )
        )

        lpm_concept_df.to_csv(
            OUT
            / "lpm_sequence_concept_inventory.csv",
            index=False,
        )


# =============================================================================
# REPORT
# =============================================================================

section("3. COURSE-SUPPORT FEATURE SUMMARY")

REPORT.append(
    feature_summary.to_string(
        index=False
    )
)

section("4. PRIMARY SEQUENCE CONCEPTS")

REPORT.append(
    sequence_concepts.to_string(
        index=False
    )
)

section("5. COURSE-AWARE TRIGGER RESULTS")

REPORT.append(
    summary_df.to_string(
        index=False
    )
)

REPORT.append("")
REPORT.append(
    "Hybrid confusion matrix:"
)

REPORT.append(
    cm_df.to_string()
)

REPORT.append("")
REPORT.append(
    classification_report(
        tasks["truth"],
        tasks[
            "hybrid_prediction"
        ],
        labels=LABELS,
        digits=4,
        zero_division=0,
    )
)

section("6. ERROR ANALYSIS")

REPORT.append(
    f"Errors: {len(errors)} / {len(tasks)}"
)

if len(errors):

    REPORT.append("")
    REPORT.append(
        "Error transitions:"
    )

    REPORT.append(
        errors.groupby(
            [
                "truth",
                "hybrid_prediction",
            ]
        )
        .size()
        .to_string()
    )

    REPORT.append("")
    REPORT.append(
        "First 30 errors:"
    )

    for _, r in errors.head(
        30
    ).iterrows():

        REPORT.append(
            f"{r['task_id']} | "
            f"true={r['truth']} | "
            f"pred={r['hybrid_prediction']} | "
            f"concept={r['concept']} | "
            f"pos=L{r['lecture']} S{r['slide']} | "
            f"past_docs={r['lex_past_docs']} | "
            f"future_docs={r['lex_future_docs']} | "
            f"sem_past={r['sem_past_max']:.4f} | "
            f"sem_future={r['sem_future_max']:.4f} | "
            f"p_seq={r['sequence_probability']} | "
            f"Q={r['question_stripped']}"
        )


section("7. EXTERNAL LPM SEQUENCE-CONCEPT INVENTORY")

if lpm_concept_df.empty:

    REPORT.append(
        "No external sequence-concept inventory available."
    )

else:

    REPORT.append(
        lpm_concept_df.to_string(
            index=False
        )
    )


section("8. SCIENTIFIC INTERPRETATION CHECKS")

REPORT.append(
    "The learned sequence detector does not receive raw question tokens or concept identity. "
    "Its inputs are only lexical/semantic evidence-availability features before and after the learner's current position."
)

REPORT.append(
    "Clinical intent is handled separately using clinical-action semantics, so a failure of sequence detection cannot be hidden inside the clinical class."
)

REPORT.append(
    "The label-free rules provide a check that any gain is not solely due to supervised threshold fitting."
)

REPORT.append(
    "If the hybrid retains high sequence recall with low false abstention, it is suitable for propagation through the deterministic verifier."
)

REPORT.append(
    "If it does not, the paper should retain oracle task-type verification only as an upper-bound diagnostic rather than presenting the verifier as directly deployable."
)

REPORT.append(
    "External LPM sequence concepts are inventoried separately because strong cross-corpus evidence requires concept diversity rather than another evaluation of identical sequence templates."
)


report_text = "\n".join(
    REPORT
)

(
    OUT
    / "COURSE_AWARE_TRIGGER_REPORT.txt"
).write_text(
    report_text,
    encoding="utf-8",
)

metadata = {
    "primary_tasks":
        len(tasks),

    "primary_corpus_slides":
        len(docs),

    "embedding_model":
        MODEL_NAME,

    "sequence_model_features":
        FEATURES,

    "sequence_model_raw_question_tokens_used":
        False,

    "sequence_model_concept_identity_used":
        False,

    "clinical_detector":
        "separate clinical-action semantic rule",

    "cv":
        "5-fold StratifiedGroupKFold grouped by lecture",

    "cuda_visible_devices":
        os.environ.get(
            "CUDA_VISIBLE_DEVICES"
        ),
}

(
    OUT
    / "course_aware_trigger_metadata.json"
).write_text(
    json.dumps(
        metadata,
        indent=2,
    ),
    encoding="utf-8",
)

print("")
print(
    report_text
)

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
    "COURSE-AWARE TRIGGER EXPERIMENT COMPLETE"
)
