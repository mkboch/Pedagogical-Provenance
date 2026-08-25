#!/usr/bin/env python3

from pathlib import Path
from collections import Counter, defaultdict
import ast
import json
import math
import os
import re
import time
import platform

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "artifacts/advanced_retrieval_review"
OUT.mkdir(parents=True, exist_ok=True)

ANSWERABLE = {
    "evidence_cited_qa",
    "slide_local_factual_qa",
    "neighbor_slide_conceptual_qa",
}

EVAL_K = [1, 3, 5, 10]
FIRST_STAGE_K = 50
LECTURE_K = [1, 3, 5]

EMBED_MODEL = "BAAI/bge-base-en-v1.5"
RERANK_MODEL = "BAAI/bge-reranker-base"

# Keep CPU demand modest because the machine is serving a website.
CPU_THREADS = min(8, os.cpu_count() or 1)

REPORT = []

def section(title):
    REPORT.append("")
    REPORT.append("=" * 100)
    REPORT.append(title)
    REPORT.append("=" * 100)


# =====================================================================
# HELPERS
# =====================================================================

def tokenize(text):
    return re.findall(r"[a-z0-9]+", str(text).lower())


def parse_listish(value):
    if value is None:
        return []

    if isinstance(value, float) and math.isnan(value):
        return []

    if isinstance(value, (list, tuple, set)):
        return [str(x) for x in value if str(x).strip()]

    s = str(value).strip()

    if not s or s.lower() in {"nan", "none", "null", "[]"}:
        return []

    try:
        obj = ast.literal_eval(s)
        if isinstance(obj, (list, tuple, set)):
            return [str(x) for x in obj if str(x).strip()]
    except Exception:
        pass

    ids = re.findall(
        r"lecture[_\-\s]*(\d+)[_\-\s]*slide[_\-\s]*(\d+)",
        s,
        flags=re.I,
    )

    if ids:
        return [
            f"lecture_{int(a):02d}_slide_{int(b):03d}"
            for a, b in ids
        ]

    return [
        x.strip().strip("'\"")
        for x in re.split(r"[;,|]", s)
        if x.strip()
    ]


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


def normalize(v):
    v = np.asarray(v, dtype=np.float32)
    n = np.linalg.norm(v)

    if n <= 0:
        return v

    return v / n


# =====================================================================
# INPUTS
# =====================================================================

DETAIL_CANDIDATES = [
    ROOT / "artifacts/paper_tables/full1000_final_statistics/full1000_all_models_metrics_detail.csv",
    ROOT / "artifacts/core_evaluation/full_task_metrics_detail.csv",
]

DETAIL = next(
    (p for p in DETAIL_CANDIDATES if p.exists()),
    None
)

if DETAIL is None:
    matches = sorted(
        ROOT.glob("**/full1000_all_models_metrics_detail.csv")
    )
    if not matches:
        raise SystemExit(
            "FAILED: full1000_all_models_metrics_detail.csv not found"
        )
    DETAIL = matches[0]

CORPUS = (
    ROOT
    / "artifacts/retrieval_inputs"
    / "discovered_slide_corpus.csv"
)

if not CORPUS.exists():
    matches = sorted(
        ROOT.glob("**/discovered_slide_corpus.csv")
    )
    if not matches:
        raise SystemExit(
            "FAILED: discovered_slide_corpus.csv not found"
        )
    CORPUS = matches[0]

raw = pd.read_csv(DETAIL)

required = [
    "task_id",
    "task_type",
    "target_doc_id",
    "gold_evidence_doc_ids",
    "question",
]

missing = [
    c for c in required
    if c not in raw.columns
]

if missing:
    raise SystemExit(
        f"FAILED: missing benchmark columns {missing}"
    )


# =====================================================================
# ANSWERABLE TASKS ONLY
# =====================================================================

task_df = (
    raw[required]
    .drop_duplicates("task_id")
    .copy()
)

task_df = task_df[
    task_df["task_type"].isin(ANSWERABLE)
].copy()

tasks = []

for _, r in task_df.iterrows():

    target = canonical_doc_id(
        r["target_doc_id"]
    )

    pos = parse_position(target)

    if pos is None:
        continue

    gold = sorted(set(
        canonical_doc_id(x)
        for x in parse_listish(
            r["gold_evidence_doc_ids"]
        )
        if str(x).strip()
    ))

    if not gold:
        gold = [target]

    tasks.append({
        "task_id": str(r["task_id"]),
        "task_type": str(r["task_type"]),
        "target_id": target,
        "target_pos": pos,
        "gold_ids": gold,
        "question": str(r["question"]),
    })

if len(tasks) != 600:
    print(
        f"WARNING: expected 600 answerable tasks, got {len(tasks)}"
    )

print("Answerable tasks:", len(tasks))
print(
    pd.Series(
        [t["task_type"] for t in tasks]
    ).value_counts().to_string()
)


# =====================================================================
# CORPUS
# =====================================================================

cdf = pd.read_csv(CORPUS)

docs_by_id = {}

for _, r in cdf.iterrows():

    did = canonical_doc_id(r["doc_id"])
    pos = parse_position(did)

    if pos is None:
        continue

    text = str(r["text"])

    item = {
        "doc_id": did,
        "lecture": pos[0],
        "slide": pos[1],
        "pos": pos,
        "text": text,
    }

    if (
        did not in docs_by_id
        or len(text) > len(
            docs_by_id[did]["text"]
        )
    ):
        docs_by_id[did] = item

docs = sorted(
    docs_by_id.values(),
    key=lambda x: (
        x["lecture"],
        x["slide"],
        x["doc_id"]
    ),
)

texts = [
    d["text"]
    for d in docs
]

doc_index = {
    d["doc_id"]: i
    for i, d in enumerate(docs)
}

print("Corpus slides:", len(docs))

lectures = sorted(set(
    d["lecture"]
    for d in docs
))

print("Lectures:", len(lectures))


# =====================================================================
# BM25
# =====================================================================

class BM25:
    def __init__(
        self,
        tokenized_docs,
        k1=1.5,
        b=0.75,
    ):

        self.docs = tokenized_docs
        self.N = len(tokenized_docs)
        self.k1 = k1
        self.b = b

        self.avgdl = (
            sum(len(x) for x in tokenized_docs)
            / max(self.N, 1)
        )

        self.tf = []
        df = Counter()

        for d in tokenized_docs:

            c = Counter(d)
            self.tf.append(c)

            for term in c:
                df[term] += 1

        self.idf = {
            term: math.log(
                1
                + (
                    self.N
                    - freq
                    + 0.5
                )
                / (
                    freq
                    + 0.5
                )
            )
            for term, freq in df.items()
        }

    def score_indices(
        self,
        query_terms,
        indices,
    ):

        result = np.zeros(
            len(indices),
            dtype=np.float32,
        )

        for jj, idx in enumerate(indices):

            c = self.tf[idx]
            dl = len(self.docs[idx])

            norm = self.k1 * (
                1
                - self.b
                + self.b
                * dl
                / max(
                    self.avgdl,
                    1e-9,
                )
            )

            score = 0.0

            for term in query_terms:

                f = c.get(term, 0)

                if f:
                    score += (
                        self.idf.get(
                            term,
                            0.0
                        )
                        * (
                            f
                            * (
                                self.k1
                                + 1
                            )
                        )
                        / (
                            f
                            + norm
                        )
                    )

            result[jj] = score

        return result


doc_tokens = [
    tokenize(x)
    for x in texts
]

print("\nBuilding BM25...")
bm25 = BM25(doc_tokens)


# =====================================================================
# TF-IDF FOR RRF
# =====================================================================

from sklearn.feature_extraction.text import TfidfVectorizer

print("Building TF-IDF...")

tfidf = TfidfVectorizer(
    lowercase=True,
    token_pattern=(
        r"(?u)\b[a-zA-Z0-9]+\b"
    ),
)

X = tfidf.fit_transform(texts)

questions = [
    t["question"]
    for t in tasks
]

Q = tfidf.transform(
    questions
)


# =====================================================================
# BGE EMBEDDINGS — CPU
# =====================================================================

import torch

torch.set_num_threads(
    CPU_THREADS
)

if torch.cuda.is_available():
    raise SystemExit(
        "SAFETY STOP: CUDA became visible."
    )

from sentence_transformers import SentenceTransformer

print(
    "\nLoading embedding model:",
    EMBED_MODEL
)

embedder = SentenceTransformer(
    EMBED_MODEL,
    device="cpu",
)

doc_embeddings = embedder.encode(
    texts,
    batch_size=32,
    normalize_embeddings=True,
    convert_to_numpy=True,
    show_progress_bar=True,
)

query_inputs = [
    "Represent this sentence for searching relevant passages: "
    + q
    for q in questions
]

query_embeddings = embedder.encode(
    query_inputs,
    batch_size=32,
    normalize_embeddings=True,
    convert_to_numpy=True,
    show_progress_bar=True,
)


# =====================================================================
# FIRST-STAGE RANKING
# =====================================================================

def candidate_indices(task, mode):

    if mode == "global":

        return np.arange(
            len(docs),
            dtype=int,
        )

    if mode == "position_constrained":

        return np.array(
            [
                i
                for i, d in enumerate(docs)
                if d["pos"]
                <= task["target_pos"]
            ],
            dtype=int,
        )

    raise ValueError(mode)


def rank_bm25(ti, inds):

    scores = bm25.score_indices(
        tokenize(tasks[ti]["question"]),
        inds,
    )

    order = np.argsort(
        -scores,
        kind="stable",
    )

    return [
        int(inds[x])
        for x in order
    ]


def rank_tfidf(ti, inds):

    scores = np.asarray(
        (
            Q[ti]
            @ X[inds].T
        ).todense()
    ).ravel()

    order = np.argsort(
        -scores,
        kind="stable",
    )

    return [
        int(inds[x])
        for x in order
    ]


def rank_rrf(ti, inds):

    rb = rank_bm25(
        ti,
        inds,
    )

    rt = rank_tfidf(
        ti,
        inds,
    )

    scores = defaultdict(float)

    for rank, idx in enumerate(
        rb,
        start=1,
    ):
        scores[idx] += (
            1.0
            / (
                60.0
                + rank
            )
        )

    for rank, idx in enumerate(
        rt,
        start=1,
    ):
        scores[idx] += (
            1.0
            / (
                60.0
                + rank
            )
        )

    return [
        idx
        for idx, _ in sorted(
            scores.items(),
            key=lambda x: (
                -x[1],
                x[0],
            ),
        )
    ]


def rank_bge(ti, inds):

    scores = (
        doc_embeddings[inds]
        @ query_embeddings[ti]
    )

    order = np.argsort(
        -scores,
        kind="stable",
    )

    return [
        int(inds[x])
        for x in order
    ]


FIRST_METHODS = {
    "bm25": rank_bm25,
    "rrf": rank_rrf,
    "bge": rank_bge,
}

rank_cache = {}

print(
    "\nBuilding first-stage rankings..."
)

start = time.time()

for ti, task in enumerate(tasks):

    if ti % 50 == 0:
        print(
            "first-stage",
            ti,
            "/",
            len(tasks),
            "elapsed",
            round(
                time.time()
                - start,
                1
            ),
            "sec",
        )

    for mode in [
        "global",
        "position_constrained",
    ]:

        inds = candidate_indices(
            task,
            mode,
        )

        for method, fn in FIRST_METHODS.items():

            rank_cache[
                (
                    ti,
                    method,
                    mode,
                )
            ] = fn(
                ti,
                inds,
            )


# =====================================================================
# METRIC FUNCTIONS
# =====================================================================

def ranking_metrics(
    task,
    ranking,
    k,
):

    top = ranking[:k]

    ids = [
        docs[i]["doc_id"]
        for i in top
    ]

    positions = [
        docs[i]["pos"]
        for i in top
    ]

    gold = set(
        task["gold_ids"]
    )

    idset = set(ids)

    return {
        "any_gold": int(
            bool(
                gold
                & idset
            )
        ),

        "target_slide": int(
            task["target_id"]
            in idset
        ),

        "target_lecture": int(
            any(
                p[0]
                == task[
                    "target_pos"
                ][0]
                for p in positions
            )
        ),
    }


def rr_metrics(
    task,
    ranking,
):

    gold = set(
        task["gold_ids"]
    )

    first_gold = None
    target_rank = None

    for rank, idx in enumerate(
        ranking,
        start=1,
    ):

        did = docs[idx]["doc_id"]

        if (
            first_gold is None
            and did in gold
        ):
            first_gold = rank

        if (
            target_rank is None
            and did
            == task["target_id"]
        ):
            target_rank = rank

        if (
            first_gold is not None
            and target_rank is not None
        ):
            break

    return {
        "rr_gold": (
            1.0
            / first_gold
            if first_gold
            else 0.0
        ),

        "rr_target": (
            1.0
            / target_rank
            if target_rank
            else 0.0
        ),
    }


# =====================================================================
# BASELINE RESULTS
# =====================================================================

baseline_rows = []

for ti, task in enumerate(tasks):

    for mode in [
        "global",
        "position_constrained",
    ]:

        for method in FIRST_METHODS:

            ranking = rank_cache[
                (
                    ti,
                    method,
                    mode,
                )
            ]

            rr = rr_metrics(
                task,
                ranking,
            )

            for k in EVAL_K:

                m = ranking_metrics(
                    task,
                    ranking,
                    k,
                )

                baseline_rows.append({
                    "task_id":
                        task["task_id"],

                    "task_type":
                        task["task_type"],

                    "method":
                        method,

                    "mode":
                        mode,

                    "k":
                        k,

                    **m,

                    **rr,
                })

baseline_detail = pd.DataFrame(
    baseline_rows
)

baseline_detail.to_csv(
    OUT
    / "baseline_unified_detail.csv",
    index=False,
)


# =====================================================================
# HIERARCHICAL BGE
#
# Stage 1:
#   represent each eligible lecture by the normalized mean of its
#   slide embeddings.
#
# Stage 2:
#   rank slides by BGE only inside top-L lectures.
#
# The position-constrained version computes lecture representations
# only from slides available at or before the learner's position.
# =====================================================================

print(
    "\nRunning hierarchical BGE..."
)

hier_rows = []

for ti, task in enumerate(tasks):

    q = query_embeddings[ti]

    for mode in [
        "global",
        "position_constrained",
    ]:

        inds = candidate_indices(
            task,
            mode,
        )

        lecture_to_inds = defaultdict(list)

        for idx in inds:
            lecture_to_inds[
                docs[idx]["lecture"]
            ].append(
                int(idx)
            )

        lecture_scores = []

        for lec, linds in lecture_to_inds.items():

            vec = normalize(
                np.mean(
                    doc_embeddings[linds],
                    axis=0,
                )
            )

            score = float(
                vec @ q
            )

            lecture_scores.append(
                (
                    lec,
                    score,
                )
            )

        lecture_scores.sort(
            key=lambda x: (
                -x[1],
                x[0],
            )
        )

        for L in LECTURE_K:

            selected_lectures = set(
                x[0]
                for x in lecture_scores[:L]
            )

            slide_inds = np.array(
                [
                    idx
                    for idx in inds
                    if docs[idx]["lecture"]
                    in selected_lectures
                ],
                dtype=int,
            )

            if len(slide_inds) == 0:
                ranking = []
            else:
                scores = (
                    doc_embeddings[slide_inds]
                    @ q
                )

                order = np.argsort(
                    -scores,
                    kind="stable",
                )

                ranking = [
                    int(
                        slide_inds[x]
                    )
                    for x in order
                ]

            rr = rr_metrics(
                task,
                ranking,
            )

            for k in EVAL_K:

                m = ranking_metrics(
                    task,
                    ranking,
                    k,
                )

                hier_rows.append({
                    "task_id":
                        task["task_id"],

                    "task_type":
                        task["task_type"],

                    "method":
                        f"hier_bge_L{L}",

                    "mode":
                        mode,

                    "k":
                        k,

                    "selected_lectures":
                        L,

                    "candidate_slide_count":
                        len(slide_inds),

                    **m,

                    **rr,
                })

hier_detail = pd.DataFrame(
    hier_rows
)

hier_detail.to_csv(
    OUT
    / "hierarchical_bge_detail.csv",
    index=False,
)


# =====================================================================
# CROSS-ENCODER CANDIDATE SETS
# =====================================================================

print(
    "\nBuilding reranker candidate sets..."
)

candidate_sets = {}

ceiling_rows = []

for ti, task in enumerate(tasks):

    for mode in [
        "global",
        "position_constrained",
    ]:

        rb = rank_cache[
            (
                ti,
                "bm25",
                mode,
            )
        ][:FIRST_STAGE_K]

        rr = rank_cache[
            (
                ti,
                "rrf",
                mode,
            )
        ][:FIRST_STAGE_K]

        rg = rank_cache[
            (
                ti,
                "bge",
                mode,
            )
        ][:FIRST_STAGE_K]

        # Standard single-retriever reranking baseline.
        bge50 = list(
            dict.fromkeys(rg)
        )

        # Strong candidate fusion:
        # union of top-50 from all three first-stage retrievers.
        union50 = list(
            dict.fromkeys(
                rg
                + rb
                + rr
            )
        )

        candidate_sets[
            (
                ti,
                mode,
                "bge50",
            )
        ] = bge50

        candidate_sets[
            (
                ti,
                mode,
                "union50",
            )
        ] = union50

        for cname, cand in [
            (
                "bge50",
                bge50,
            ),
            (
                "union50",
                union50,
            ),
        ]:

            ids = set(
                docs[i]["doc_id"]
                for i in cand
            )

            gold = set(
                task["gold_ids"]
            )

            ceiling_rows.append({
                "task_id":
                    task["task_id"],

                "task_type":
                    task["task_type"],

                "mode":
                    mode,

                "candidate_source":
                    cname,

                "candidate_count":
                    len(cand),

                "candidate_any_gold":
                    int(
                        bool(
                            gold
                            & ids
                        )
                    ),

                "candidate_target_slide":
                    int(
                        task["target_id"]
                        in ids
                    ),
            })

ceiling_df = pd.DataFrame(
    ceiling_rows
)

ceiling_df.to_csv(
    OUT
    / "reranker_candidate_ceiling_detail.csv",
    index=False,
)


# =====================================================================
# BUILD UNIQUE QUERY-DOCUMENT PAIRS
# =====================================================================

pair_keys = set()

for (
    ti,
    mode,
    cname,
), cand in candidate_sets.items():

    for idx in cand:
        pair_keys.add(
            (
                ti,
                idx,
            )
        )

pair_keys = sorted(
    pair_keys
)

print(
    "Unique query-document pairs to cross-encode:",
    len(pair_keys)
)


# =====================================================================
# LOAD CROSS ENCODER ON CPU
# =====================================================================

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
)

print(
    "\nLoading reranker:",
    RERANK_MODEL
)

rerank_tokenizer = (
    AutoTokenizer
    .from_pretrained(
        RERANK_MODEL
    )
)

reranker = (
    AutoModelForSequenceClassification
    .from_pretrained(
        RERANK_MODEL
    )
)

reranker.to("cpu")
reranker.eval()

if torch.cuda.is_available():
    raise SystemExit(
        "SAFETY STOP: CUDA became visible after reranker load."
    )


# =====================================================================
# CROSS-ENCODER SCORING
# =====================================================================

BATCH = 16

scores = {}

print(
    "\nCross-encoder scoring begins."
)
print(
    "This is deliberately CPU-only and low priority."
)
print(
    "It can take a while; progress will print every 100 batches."
)

n_batches = math.ceil(
    len(pair_keys)
    / BATCH
)

start = time.time()

with torch.inference_mode():

    for batch_no, start_idx in enumerate(
        range(
            0,
            len(pair_keys),
            BATCH,
        ),
        start=1,
    ):

        batch_keys = pair_keys[
            start_idx:
            start_idx
            + BATCH
        ]

        qbatch = [
            tasks[ti]["question"]
            for ti, _ in batch_keys
        ]

        dbatch = [
            docs[idx]["text"]
            for _, idx in batch_keys
        ]

        tok = rerank_tokenizer(
            qbatch,
            dbatch,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )

        out = reranker(
            **tok
        )

        logits = (
            out.logits
            .view(-1)
            .float()
            .cpu()
            .numpy()
        )

        for key, score in zip(
            batch_keys,
            logits,
        ):
            scores[key] = float(
                score
            )

        if (
            batch_no == 1
            or batch_no % 100 == 0
            or batch_no == n_batches
        ):

            elapsed = (
                time.time()
                - start
            )

            rate = (
                start_idx
                + len(batch_keys)
            ) / max(
                elapsed,
                1e-9,
            )

            remain = (
                len(pair_keys)
                - (
                    start_idx
                    + len(batch_keys)
                )
            )

            eta = (
                remain
                / max(
                    rate,
                    1e-9,
                )
            )

            print(
                f"reranker batch "
                f"{batch_no}/{n_batches} | "
                f"pairs "
                f"{start_idx + len(batch_keys)}/"
                f"{len(pair_keys)} | "
                f"elapsed={elapsed/60:.1f} min | "
                f"ETA={eta/60:.1f} min"
            )


# =====================================================================
# RERANK
# =====================================================================

rerank_rows = []

ranking_exports = []

for (
    ti,
    mode,
    cname,
), cand in candidate_sets.items():

    ranking = sorted(
        cand,
        key=lambda idx: (
            -scores[
                (
                    ti,
                    idx,
                )
            ],
            idx,
        ),
    )

    task = tasks[ti]

    rr = rr_metrics(
        task,
        ranking,
    )

    ranking_exports.append({
        "task_id":
            task["task_id"],

        "task_type":
            task["task_type"],

        "mode":
            mode,

        "candidate_source":
            cname,

        "ranked_doc_ids":
            json.dumps(
                [
                    docs[i]["doc_id"]
                    for i in ranking
                ]
            ),

        "ranked_scores":
            json.dumps(
                [
                    scores[
                        (
                            ti,
                            i,
                        )
                    ]
                    for i in ranking
                ]
            ),
    })

    for k in EVAL_K:

        m = ranking_metrics(
            task,
            ranking,
            k,
        )

        rerank_rows.append({
            "task_id":
                task["task_id"],

            "task_type":
                task["task_type"],

            "method":
                f"bge_reranker_{cname}",

            "mode":
                mode,

            "k":
                k,

            "candidate_count":
                len(cand),

            **m,

            **rr,
        })


rerank_detail = pd.DataFrame(
    rerank_rows
)

rerank_detail.to_csv(
    OUT
    / "cross_encoder_reranker_detail.csv",
    index=False,
)

pd.DataFrame(
    ranking_exports
).to_csv(
    OUT
    / "cross_encoder_rankings.csv",
    index=False,
)


# =====================================================================
# AGGREGATION HELPERS
# =====================================================================

def aggregate(df, extra_cols=None):

    extra_cols = (
        extra_cols or []
    )

    group_cols = (
        [
            "method",
            "mode",
            "k",
        ]
        + extra_cols
    )

    overall = (
        df.groupby(
            group_cols
        )
        .agg(
            N=("task_id", "count"),
            AnyGold=(
                "any_gold",
                "mean"
            ),
            TargetSlide=(
                "target_slide",
                "mean"
            ),
            TargetLecture=(
                "target_lecture",
                "mean"
            ),
            MRRAnyGold=(
                "rr_gold",
                "mean"
            ),
            MRRTargetSlide=(
                "rr_target",
                "mean"
            ),
        )
        .reset_index()
    )

    overall["task_scope"] = "ALL_ANSWERABLE"

    by_task = (
        df.groupby(
            group_cols
            + [
                "task_type"
            ]
        )
        .agg(
            N=("task_id", "count"),
            AnyGold=(
                "any_gold",
                "mean"
            ),
            TargetSlide=(
                "target_slide",
                "mean"
            ),
            TargetLecture=(
                "target_lecture",
                "mean"
            ),
            MRRAnyGold=(
                "rr_gold",
                "mean"
            ),
            MRRTargetSlide=(
                "rr_target",
                "mean"
            ),
        )
        .reset_index()
        .rename(
            columns={
                "task_type":
                    "task_scope"
            }
        )
    )

    return pd.concat(
        [
            overall,
            by_task,
        ],
        ignore_index=True,
    )


baseline_summary = aggregate(
    baseline_detail
)

hier_summary = aggregate(
    hier_detail
)

rerank_summary = aggregate(
    rerank_detail
)

baseline_summary.to_csv(
    OUT
    / "baseline_unified_summary.csv",
    index=False,
)

hier_summary.to_csv(
    OUT
    / "hierarchical_bge_summary.csv",
    index=False,
)

rerank_summary.to_csv(
    OUT
    / "cross_encoder_reranker_summary.csv",
    index=False,
)


# =====================================================================
# CANDIDATE CEILING SUMMARY
# =====================================================================

ceiling_all = (
    ceiling_df.groupby(
        [
            "mode",
            "candidate_source",
        ]
    )
    .agg(
        N=("task_id", "count"),
        MeanCandidateCount=(
            "candidate_count",
            "mean"
        ),
        AnyGoldCeiling=(
            "candidate_any_gold",
            "mean"
        ),
        TargetSlideCeiling=(
            "candidate_target_slide",
            "mean"
        ),
    )
    .reset_index()
)

ceiling_task = (
    ceiling_df.groupby(
        [
            "mode",
            "candidate_source",
            "task_type",
        ]
    )
    .agg(
        N=("task_id", "count"),
        MeanCandidateCount=(
            "candidate_count",
            "mean"
        ),
        AnyGoldCeiling=(
            "candidate_any_gold",
            "mean"
        ),
        TargetSlideCeiling=(
            "candidate_target_slide",
            "mean"
        ),
    )
    .reset_index()
)

ceiling_all.to_csv(
    OUT
    / "reranker_candidate_ceiling_summary.csv",
    index=False,
)

ceiling_task.to_csv(
    OUT
    / "reranker_candidate_ceiling_by_task.csv",
    index=False,
)


# =====================================================================
# MASTER COMPARISON
# =====================================================================

# k=3 is the practical context depth used in the paper.
base_k3 = baseline_summary[
    (
        baseline_summary["k"]
        == 3
    )
    &
    (
        baseline_summary["task_scope"]
        == "ALL_ANSWERABLE"
    )
].copy()

hier_k3 = hier_summary[
    (
        hier_summary["k"]
        == 3
    )
    &
    (
        hier_summary["task_scope"]
        == "ALL_ANSWERABLE"
    )
].copy()

rerank_k3 = rerank_summary[
    (
        rerank_summary["k"]
        == 3
    )
    &
    (
        rerank_summary["task_scope"]
        == "ALL_ANSWERABLE"
    )
].copy()

master = pd.concat(
    [
        base_k3,
        hier_k3,
        rerank_k3,
    ],
    ignore_index=True,
    sort=False,
)

master = master[
    [
        "method",
        "mode",
        "N",
        "AnyGold",
        "TargetSlide",
        "TargetLecture",
        "MRRAnyGold",
        "MRRTargetSlide",
    ]
]

master.to_csv(
    OUT
    / "paper_candidate_top3_comparison.csv",
    index=False,
)


# =====================================================================
# REPORT
# =====================================================================

section(
    "ADVANCED RETRIEVAL EXPERIMENT"
)

REPORT.append(
    f"Answerable tasks: {len(tasks)}"
)

REPORT.append(
    f"Corpus slides: {len(docs)}"
)

REPORT.append(
    f"Embedding model: {EMBED_MODEL}"
)

REPORT.append(
    f"Reranker: {RERANK_MODEL}"
)

REPORT.append(
    f"Reranker unique query-document pairs: {len(pair_keys)}"
)

section(
    "1. TOP-50 CANDIDATE CEILINGS"
)

REPORT.append(
    ceiling_all.to_string(
        index=False
    )
)

section(
    "2. PRACTICAL TOP-3 COMPARISON"
)

REPORT.append(
    master.sort_values(
        [
            "mode",
            "AnyGold",
        ],
        ascending=[
            True,
            False,
        ],
    ).to_string(
        index=False
    )
)

section(
    "3. HIERARCHICAL BGE — TOP-3"
)

REPORT.append(
    hier_k3[
        [
            "method",
            "mode",
            "N",
            "AnyGold",
            "TargetSlide",
            "TargetLecture",
            "MRRAnyGold",
            "MRRTargetSlide",
        ]
    ].to_string(
        index=False
    )
)

section(
    "4. CROSS-ENCODER RERANKING — ALL k"
)

REPORT.append(
    rerank_summary[
        rerank_summary[
            "task_scope"
        ].eq(
            "ALL_ANSWERABLE"
        )
    ][
        [
            "method",
            "mode",
            "k",
            "N",
            "AnyGold",
            "TargetSlide",
            "TargetLecture",
            "MRRAnyGold",
            "MRRTargetSlide",
        ]
    ].to_string(
        index=False
    )
)

section(
    "5. INTERPRETATION RULES"
)

REPORT.append(
    "If the candidate ceiling is high but reranked Recall@3 remains low, "
    "candidate generation is not the main bottleneck; ranking remains difficult."
)

REPORT.append(
    "If reranking substantially raises Recall@3, the paper should describe "
    "the original failure as a ranking/localization problem rather than evidence being unrecoverable."
)

REPORT.append(
    "If hierarchical BGE raises target-slide Recall@3, course hierarchy itself provides useful retrieval structure."
)

REPORT.append(
    "For position-constrained neighbor-slide QA, AllGold is intentionally not used as a primary metric because future adjacent evidence may lie beyond the learner's current position."
)

report_text = "\n".join(
    REPORT
)

(
    OUT
    / "ADVANCED_RETRIEVAL_REPORT.txt"
).write_text(
    report_text,
    encoding="utf-8",
)

metadata = {
    "answerable_tasks":
        len(tasks),

    "corpus_slides":
        len(docs),

    "embedding_model":
        EMBED_MODEL,

    "reranker_model":
        RERANK_MODEL,

    "first_stage_k":
        FIRST_STAGE_K,

    "evaluation_k":
        EVAL_K,

    "hierarchical_lecture_k":
        LECTURE_K,

    "cpu_threads":
        CPU_THREADS,

    "cuda_visible_devices":
        os.environ.get(
            "CUDA_VISIBLE_DEVICES"
        ),

    "python":
        platform.python_version(),
}

(
    OUT
    / "advanced_retrieval_metadata.json"
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
    "ADVANCED RETRIEVAL EXPERIMENT COMPLETE"
)
