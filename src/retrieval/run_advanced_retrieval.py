#!/usr/bin/env python3
"""
ADVANCED RETRIEVAL UNDER THE AUTHORITATIVE POSITION-CONSTRAINED PROTOCOL
===========================================================================

Purpose
-------
Re-run advanced retrieval (BGE dense, hierarchical lecture->slide, cross-encoder
reranking) under the AUTHORITATIVE reference protocol protocol:

  * 1,117-document corpus (data/corpus/slide_corpus_final.jsonl)
  * reference BM25 / TF-IDF / RRF implementations recovered by AST from the
    reference full1000 pipeline
  * per-task course-position eligibility mask (course_position <= target's
    course_position), computed BEFORE ranking and WITHOUT using gold/target
    text or gold identity in any scoring
  * the fixed 600 answerable tasks (evidence_cited_qa, slide_local_factual_qa,
    neighbor_slide_conceptual_qa) from core_qa_benchmark.jsonl

The earlier exploratory script scripts/run_advanced_retrieval_review.py used a
STALE 1,112-slide corpus and a different eligibility rule; its numbers are NOT
comparable to the reference protocol. This script reuses only its *modeling* code
(BGE encoding, lecture aggregation, cross-encoder reranking, candidate-pool
construction) and rebuilds everything else on the authoritative protocol.

HARD SELF-VERIFICATION GATE
---------------------------
Our from-scratch BM25 and RRF must exactly reproduce the frozen authoritative
answerable-task values:
    BM25 AnyGoldRecall@3 = 0.090000
    RRF  AnyGoldRecall@3 = 0.145000
(source: artifacts/matched_retrieval/
         position_retrieval_k_sweep_answerable.csv)
If the gate fails the script aborts before any dense/rerank work.

CPU-ONLY. No LLM text generation. Nothing in the read-only reference
directories is written.
"""

from pathlib import Path
from collections import defaultdict
import argparse
import ast
import json
import math
import os
import platform
import re
import time

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


# ============================================================================
# Configuration
# ============================================================================

ROOT = Path(__file__).resolve().parents[2]

# Reference pipeline holding the historical helper implementations.
PIPE = ROOT / "src/common/runtime_reference.py"

CORPUS = ROOT / "data/corpus/slide_corpus_final.jsonl"
BENCH = ROOT / "data/benchmark/core_qa_benchmark.jsonl"

# Read-only cross-check reference (authoritative task identity + gate values).
FREEZE_DIR = ROOT / "artifacts/matched_retrieval"
FROZEN_CSV = FREEZE_DIR / "frozen_position_contexts_and_prompts_2000.csv"
FREEZE_KSWEEP = FREEZE_DIR / "position_retrieval_k_sweep_answerable.csv"

OUT = ROOT / "artifacts/retrieval_diagnostics"
OUT.mkdir(parents=True, exist_ok=True)

ANSWERABLE = {
    "evidence_cited_qa",
    "neighbor_slide_conceptual_qa",
    "slide_local_factual_qa",
}

COMPONENT_TOP_K = 50          # reference component truncation
K_SWEEP = [1, 3, 5, 10, 20, 50]
LECTURE_K = [1, 3, 5]         # hierarchical stage-1 depth (from prior script)
FIRST_STAGE_K = 50            # reranker candidate pool depth (prior script)

EMBED_MODEL = "BAAI/bge-base-en-v1.5"
RERANK_MODEL = "BAAI/bge-reranker-base"

BGE_QUERY_PREFIX = (
    "Represent this sentence for searching relevant passages: "
)

CPU_THREADS = int(
    os.environ.get(
        "PEDPROV_CPU_THREADS",
        str(min(16, os.cpu_count() or 1)),
    )
)

RERANK_BATCH = 32

GATE_BM25_K3 = 0.090000
GATE_RRF_K3 = 0.145000
GATE_TOL = 1e-9


# ============================================================================
# Utilities
# ============================================================================

def read_jsonl(path):
    return [
        json.loads(line)
        for line in path.open("r", encoding="utf-8")
        if line.strip()
    ]


def recover_reference_helpers(path):
    """
    Same AST-recovery pattern the reference protocol script uses on this same reference
    source. We only need the retrieval helpers (no context formatting or
    prompt building, since this run performs no generation).
    """
    text = path.read_text(encoding="utf-8", errors="replace")
    tree = ast.parse(text)

    wanted = {
        "norm",
        "tokenize",
        "unique",
        "doc_text",
        "simple_bm25_scores",
        "rrf",
    }

    body = []

    for node in tree.body:
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            try:
                ast.literal_eval(node.value)
            except Exception:
                continue
            body.append(node)

    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in wanted:
            body.append(node)

    mod = ast.Module(body=body, type_ignores=[])
    ast.fix_missing_locations(mod)

    ns = {
        "re": re,
        "json": json,
        "np": np,
        "math": math,
        "defaultdict": defaultdict,
    }

    exec(compile(mod, str(path), "exec"), ns, ns)

    for name in ["tokenize", "doc_text", "simple_bm25_scores", "rrf"]:
        if name not in ns:
            raise SystemExit(f"FAILED: could not recover {name}")

    return ns


def pos_value(doc):
    p = doc.get("course_position")

    if p is None or str(p) == "":
        raise RuntimeError(
            f"Missing course_position for {doc.get('doc_id')}"
        )

    return int(p)


def first_rank(ranking, relevant):
    rel = set(relevant)

    for i, d in enumerate(ranking, start=1):
        if d in rel:
            return i

    return None


def normalize_vec(v):
    v = np.asarray(v, dtype=np.float32)
    n = np.linalg.norm(v)

    if n <= 0:
        return v

    return v / n


def log(*a):
    print(*a, flush=True)


# ============================================================================
# Load authoritative data
# ============================================================================

parser = argparse.ArgumentParser()

parser.add_argument(
    "--stage",
    choices=["gate", "full"],
    default="full",
    help="'gate' stops after the BM25/RRF self-verification gate.",
)

args = parser.parse_args()

for p in [PIPE, CORPUS, BENCH, FROZEN_CSV, FREEZE_KSWEEP]:
    if not p.exists():
        raise SystemExit(f"FAILED: missing {p}")

corpus = read_jsonl(CORPUS)
all_tasks = read_jsonl(BENCH)

if len(corpus) != 1117:
    raise SystemExit(
        f"FAILED: expected 1117 corpus docs, got {len(corpus)}"
    )

if len(all_tasks) != 1000:
    raise SystemExit(
        f"FAILED: expected 1000 benchmark tasks, got {len(all_tasks)}"
    )

corpus_by_doc = {str(x["doc_id"]): x for x in corpus}
doc_ids = [str(x["doc_id"]) for x in corpus]
doc_index = {d: i for i, d in enumerate(doc_ids)}

ns = recover_reference_helpers(PIPE)

tokenize = ns["tokenize"]
doc_text = ns["doc_text"]
simple_bm25_scores = ns["simple_bm25_scores"]
rrf = ns["rrf"]

# Lexical document representation: EXACT reference doc_text().
docs_lexical = [doc_text(x) for x in corpus]
doc_tokens = [tokenize(x) for x in docs_lexical]

# Dense/cross-encoder document representation: raw slide text, faithful to
# scripts/run_advanced_retrieval_review.py (which encoded the plain text
# column). This is the representation most favourable to the neural models
# because it omits the "Lecture N, Slide M. Document ID: ..." boilerplate.
docs_dense = [str(x.get("text", "")) for x in corpus]

doc_lecture = [str(x["lecture_id"]) for x in corpus]
doc_position = [pos_value(x) for x in corpus]


# ---------------------------------------------------------------------------
# Fixed 600 answerable tasks, identity cross-checked against the reference protocol.
# ---------------------------------------------------------------------------

tasks = [t for t in all_tasks if t["task_type"] in ANSWERABLE]

if len(tasks) != 600:
    raise SystemExit(
        f"FAILED: expected 600 answerable tasks, got {len(tasks)}"
    )

frozen = pd.read_csv(FROZEN_CSV)

frozen_ids = set(
    frozen.loc[frozen["task_type"].isin(ANSWERABLE), "task_id"]
    .astype(str)
)

our_ids = set(str(t["task_id"]) for t in tasks)

TASK_IDENTITY_MATCH = frozen_ids == our_ids

if not TASK_IDENTITY_MATCH:
    raise SystemExit(
        "FAILED: 600-task identity does not match the reference protocol "
        f"(ours={len(our_ids)}, frozen={len(frozen_ids)}, "
        f"symmetric_difference={len(our_ids ^ frozen_ids)})"
    )

questions = [str(t["question"]) for t in tasks]

log("=" * 100)
log("ADVANCED RETRIEVAL (authoritative position-constrained protocol)")
log("=" * 100)
log("Corpus docs:", len(corpus))
log("Benchmark tasks:", len(all_tasks))
log("Answerable tasks:", len(tasks))
log("Task identity matches reference protocol:", TASK_IDENTITY_MATCH)
log("CPU threads:", CPU_THREADS)
log("CUDA_VISIBLE_DEVICES:", repr(os.environ.get("CUDA_VISIBLE_DEVICES")))


# ============================================================================
# Global TF-IDF (exact reference configuration, fitted on the full corpus)
# ============================================================================

tfidf_global = TfidfVectorizer(
    lowercase=True,
    stop_words="english",
    ngram_range=(1, 3),
    max_df=0.85,
)

X_global = tfidf_global.fit_transform(docs_lexical)
Q_global = tfidf_global.transform(questions)
TF_GLOBAL = Q_global @ X_global.T


# ============================================================================
# Per-task eligibility + lexical rankings (exact position-constrained semantics)
#
#   1. Score the FULL 1,117-doc corpus with the reference BM25 / global TF-IDF.
#   2. Drop documents whose course_position is AFTER the target's position.
#   3. Take the first 50 eligible BM25 docs and first 50 eligible TF-IDF docs.
#   4. Fuse with the reference RRF, truncate to 50.
#
# Eligibility uses only the task's course position; no gold/target text is
# used for scoring, and the target is never injected into the pool.
# ============================================================================

eligible_sets = {}
eligible_lists = {}
rank_store = {}
task_meta = {}

log("\nBuilding lexical rankings ...")
t0 = time.time()

for ti, task in enumerate(tasks):

    tid = str(task["task_id"])
    target = str(task["target_doc_id"])

    if target not in corpus_by_doc:
        raise SystemExit(f"FAILED: target missing from corpus: {tid} {target}")

    target_position = pos_value(corpus_by_doc[target])

    eligible_idx = [
        j for j in range(len(corpus))
        if doc_position[j] <= target_position
    ]

    eligible_set = {doc_ids[j] for j in eligible_idx}

    eligible_sets[tid] = eligible_set
    eligible_lists[tid] = eligible_idx

    gold = [str(x) for x in task.get("evidence_doc_ids", [])]

    task_meta[tid] = {
        "task_id": tid,
        "task_type": str(task["task_type"]),
        "target": target,
        "target_position": target_position,
        "target_lecture": str(corpus_by_doc[target]["lecture_id"]),
        "gold": gold,
        "gold_set": set(gold),
        "eligible_count": len(eligible_set),
        "eligible_contains_target": int(target in eligible_set),
        "eligible_contains_any_gold": int(
            bool(set(gold) & eligible_set)
        ),
        "eligible_contains_all_gold": int(
            bool(gold) and set(gold).issubset(eligible_set)
        ),
    }

    # ---- BM25 (reference scorer, full-corpus statistics) ----
    bm_scores = simple_bm25_scores(
        tokenize(task["question"]),
        doc_tokens,
    )

    bm_order_full = np.argsort(bm_scores)[::-1]

    bm_rank_eligible_full = [
        doc_ids[j] for j in bm_order_full if doc_ids[j] in eligible_set
    ]

    bm_rank50 = bm_rank_eligible_full[:COMPONENT_TOP_K]

    # ---- TF-IDF (reference global fit) ----
    tf_row = TF_GLOBAL[ti].toarray().ravel()
    tf_order_full = np.argsort(tf_row)[::-1]

    tf_rank_eligible_full = [
        doc_ids[j] for j in tf_order_full if doc_ids[j] in eligible_set
    ]

    tf_rank50 = tf_rank_eligible_full[:COMPONENT_TOP_K]

    # ---- RRF ----
    rrf_rank = rrf([bm_rank50, tf_rank50])[:COMPONENT_TOP_K]

    # position-constrained convention: BM25 keeps the complete eligible ordering; historical RRF
    # is only ever constructed through rank 50.
    rank_store[(tid, "bm25")] = bm_rank_eligible_full
    rank_store[(tid, "rrf")] = rrf_rank
    rank_store[(tid, "tfidf")] = tf_rank_eligible_full

    if ti % 100 == 0:
        log(
            f"  lexical {ti}/{len(tasks)} "
            f"elapsed={time.time() - t0:.1f}s"
        )

log(f"Lexical rankings done in {time.time() - t0:.1f}s")


# ============================================================================
# Metric machinery
# ============================================================================

def k_sweep_rows(method, ranking_by_task, ks=K_SWEEP):
    rows = []

    for tid, meta in task_meta.items():

        ranking = ranking_by_task[tid]

        for k in ks:

            top = ranking[:k]
            top_set = set(top)

            rows.append({
                "task_id": tid,
                "task_type": meta["task_type"],
                "method": method,
                "k": k,
                "any_gold": int(bool(top_set & meta["gold_set"])),
                "all_gold": int(
                    bool(meta["gold_set"])
                    and meta["gold_set"].issubset(top_set)
                ),
                "target_slide": int(meta["target"] in top_set),
                "target_lecture": int(
                    any(
                        str(corpus_by_doc[d]["lecture_id"])
                        == meta["target_lecture"]
                        for d in top
                    )
                ),
                "ranking_length": len(ranking),
            })

    return rows


def mrr_rows(method, ranking_by_task):
    rows = []

    for tid, meta in task_meta.items():

        ranking = ranking_by_task[tid]

        fg = first_rank(ranking, meta["gold"])
        ft = first_rank(ranking, [meta["target"]])

        rows.append({
            "task_id": tid,
            "task_type": meta["task_type"],
            "method": method,
            "first_gold_rank": fg,
            "first_target_rank": ft,
            "rr_any_gold": (1.0 / fg if fg else 0.0),
            "rr_target": (1.0 / ft if ft else 0.0),
            "gold_found": int(fg is not None),
            "target_found": int(ft is not None),
            "ranking_length": len(ranking),
        })

    return rows


all_ksweep = []
all_mrr = []


def register(method, ranking_by_task, ks=K_SWEEP):
    all_ksweep.extend(k_sweep_rows(method, ranking_by_task, ks))
    all_mrr.extend(mrr_rows(method, ranking_by_task))


register("bm25", {tid: rank_store[(tid, "bm25")] for tid in task_meta})
register("rrf", {tid: rank_store[(tid, "rrf")] for tid in task_meta})


# ============================================================================
# HARD SELF-VERIFICATION GATE
# ============================================================================

gate_df = pd.DataFrame(all_ksweep)

gate_bm25 = float(
    gate_df.query("method == 'bm25' and k == 3")["any_gold"].mean()
)

gate_rrf = float(
    gate_df.query("method == 'rrf' and k == 3")["any_gold"].mean()
)

authoritative = pd.read_csv(FREEZE_KSWEEP)

auth_bm25 = float(
    authoritative.query("method == 'bm25' and k == 3")
    ["AnyGoldRecall"].iloc[0]
)

auth_rrf = float(
    authoritative.query("method == 'rrf' and k == 3")
    ["AnyGoldRecall"].iloc[0]
)

bm25_ok = abs(gate_bm25 - GATE_BM25_K3) < GATE_TOL
rrf_ok = abs(gate_rrf - GATE_RRF_K3) < GATE_TOL

# Secondary (not gating) consistency check across the full k sweep.
secondary = []

for method in ["bm25", "rrf"]:
    for k in K_SWEEP:
        ours = gate_df.query("method == @method and k == @k")
        auth = authoritative.query("method == @method and k == @k")

        secondary.append({
            "method": method,
            "k": k,
            "ours_AnyGoldRecall": float(ours["any_gold"].mean()),
            "auth_AnyGoldRecall": float(auth["AnyGoldRecall"].iloc[0]),
            "ours_TargetSlideRecall": float(ours["target_slide"].mean()),
            "auth_TargetSlideRecall": float(auth["TargetSlideRecall"].iloc[0]),
            "ours_TargetLectureRecall": float(ours["target_lecture"].mean()),
            "auth_TargetLectureRecall": float(
                auth["TargetLectureRecall"].iloc[0]
            ),
        })

secondary_df = pd.DataFrame(secondary)

secondary_df["any_gold_match"] = np.isclose(
    secondary_df["ours_AnyGoldRecall"],
    secondary_df["auth_AnyGoldRecall"],
    atol=1e-12,
)

secondary_df["target_slide_match"] = np.isclose(
    secondary_df["ours_TargetSlideRecall"],
    secondary_df["auth_TargetSlideRecall"],
    atol=1e-12,
)

secondary_df["target_lecture_match"] = np.isclose(
    secondary_df["ours_TargetLectureRecall"],
    secondary_df["auth_TargetLectureRecall"],
    atol=1e-12,
)

FULL_SWEEP_MATCH = bool(
    secondary_df[
        ["any_gold_match", "target_slide_match", "target_lecture_match"]
    ].all().all()
)

secondary_df.to_csv(
    OUT / "advanced_retrieval_gate_full_sweep_check.csv",
    index=False,
)

GATE_PASS = bool(bm25_ok and rrf_ok)

log("")
log("=" * 100)
log("HARD SELF-VERIFICATION GATE")
log("=" * 100)
log(f"BM25 AnyGoldRecall@3 : ours={gate_bm25:.6f} "
    f"required={GATE_BM25_K3:.6f} frozen_csv={auth_bm25:.6f} "
    f"-> {'PASS' if bm25_ok else 'FAIL'}")
log(f"RRF  AnyGoldRecall@3 : ours={gate_rrf:.6f} "
    f"required={GATE_RRF_K3:.6f} frozen_csv={auth_rrf:.6f} "
    f"-> {'PASS' if rrf_ok else 'FAIL'}")
log(f"Full k-sweep agreement with frozen CSV (all k, all 3 recalls): "
    f"{FULL_SWEEP_MATCH}")
log(f"GATE_PASS = {GATE_PASS}")

gate_record = {
    "gate_pass": GATE_PASS,
    "bm25_any_gold_recall_at_3_ours": gate_bm25,
    "bm25_any_gold_recall_at_3_required": GATE_BM25_K3,
    "bm25_any_gold_recall_at_3_frozen_csv": auth_bm25,
    "rrf_any_gold_recall_at_3_ours": gate_rrf,
    "rrf_any_gold_recall_at_3_required": GATE_RRF_K3,
    "rrf_any_gold_recall_at_3_frozen_csv": auth_rrf,
    "full_k_sweep_match": FULL_SWEEP_MATCH,
    "task_identity_match_with_freeze": TASK_IDENTITY_MATCH,
    "n_tasks": len(tasks),
    "n_corpus_docs": len(corpus),
}

(OUT / "advanced_retrieval_gate.json").write_text(
    json.dumps(gate_record, indent=2),
    encoding="utf-8",
)

if not GATE_PASS:
    raise SystemExit(
        "STOP: self-verification gate FAILED. Not proceeding to "
        "dense/hierarchical/reranker measurement."
    )

if args.stage == "gate":
    log("\nStage 'gate' requested; stopping after gate.")
    raise SystemExit(0)


# ============================================================================
# BGE dense retrieval (CPU)
# ============================================================================

import torch

torch.set_num_threads(CPU_THREADS)

if torch.cuda.is_available():
    raise SystemExit("SAFETY STOP: CUDA is visible; this must be CPU-only.")

from sentence_transformers import SentenceTransformer

log("\nLoading embedding model:", EMBED_MODEL)

embedder = SentenceTransformer(EMBED_MODEL, device="cpu")

t0 = time.time()

doc_embeddings = embedder.encode(
    docs_dense,
    batch_size=32,
    normalize_embeddings=True,
    convert_to_numpy=True,
    show_progress_bar=False,
)

log(f"Encoded {len(docs_dense)} docs in {time.time() - t0:.1f}s")

t0 = time.time()

query_embeddings = embedder.encode(
    [BGE_QUERY_PREFIX + q for q in questions],
    batch_size=32,
    normalize_embeddings=True,
    convert_to_numpy=True,
    show_progress_bar=False,
)

log(f"Encoded {len(questions)} queries in {time.time() - t0:.1f}s")

del embedder


bge_rankings = {}
bge_rank_idx = {}

for ti, task in enumerate(tasks):

    tid = str(task["task_id"])
    inds = np.array(eligible_lists[tid], dtype=int)

    scores = doc_embeddings[inds] @ query_embeddings[ti]
    order = np.argsort(-scores, kind="stable")

    ranked_idx = [int(inds[x]) for x in order]

    bge_rank_idx[tid] = ranked_idx
    bge_rankings[tid] = [doc_ids[j] for j in ranked_idx]

register("bge_dense", bge_rankings)

log("BGE dense rankings built.")


# ============================================================================
# Hierarchical lecture -> slide retrieval
#
# Adapted from scripts/run_advanced_retrieval_review.py:
#   Stage 1: each eligible lecture is represented by the L2-normalised mean of
#            its ELIGIBLE slide embeddings; lectures ranked by cosine to query.
#   Stage 2: slides inside the top-L lectures ranked by BGE cosine.
# ============================================================================

hier_rankings = {L: {} for L in LECTURE_K}

for ti, task in enumerate(tasks):

    tid = str(task["task_id"])
    q = query_embeddings[ti]
    inds = eligible_lists[tid]

    lecture_to_inds = defaultdict(list)

    for idx in inds:
        lecture_to_inds[doc_lecture[idx]].append(int(idx))

    lecture_scores = []

    for lec, linds in lecture_to_inds.items():
        vec = normalize_vec(np.mean(doc_embeddings[linds], axis=0))
        lecture_scores.append((lec, float(vec @ q)))

    lecture_scores.sort(key=lambda x: (-x[1], x[0]))

    for L in LECTURE_K:

        selected = set(x[0] for x in lecture_scores[:L])

        slide_inds = np.array(
            [idx for idx in inds if doc_lecture[idx] in selected],
            dtype=int,
        )

        if len(slide_inds) == 0:
            hier_rankings[L][tid] = []
            continue

        scores = doc_embeddings[slide_inds] @ q
        order = np.argsort(-scores, kind="stable")

        hier_rankings[L][tid] = [
            doc_ids[int(slide_inds[x])] for x in order
        ]

for L in LECTURE_K:
    register(f"hier_bge_L{L}", hier_rankings[L])

log("Hierarchical BGE rankings built.")


# ============================================================================
# Cross-encoder candidate pools (position-constrained)
#
#   bge50   : BGE top-50 over the eligible set (single-retriever baseline)
#   union50 : dedup(BGE top-50 + BM25 top-50 + RRF top-50) over the eligible
#             set (candidate fusion; the prior script's "union50")
# ============================================================================

candidate_sets = {}
ceiling_rows = []

for ti, task in enumerate(tasks):

    tid = str(task["task_id"])
    meta = task_meta[tid]

    rg = bge_rank_idx[tid][:FIRST_STAGE_K]

    rb = [
        doc_index[d]
        for d in rank_store[(tid, "bm25")][:FIRST_STAGE_K]
    ]

    rr = [
        doc_index[d]
        for d in rank_store[(tid, "rrf")][:FIRST_STAGE_K]
    ]

    bge50 = list(dict.fromkeys(rg))
    union50 = list(dict.fromkeys(rg + rb + rr))

    candidate_sets[(ti, "bge50")] = bge50
    candidate_sets[(ti, "union50")] = union50

    for cname, cand in [("bge50", bge50), ("union50", union50)]:

        ids = set(doc_ids[i] for i in cand)

        ceiling_rows.append({
            "task_id": tid,
            "task_type": meta["task_type"],
            "candidate_source": cname,
            "candidate_count": len(cand),
            "candidate_any_gold": int(bool(meta["gold_set"] & ids)),
            "candidate_all_gold": int(
                bool(meta["gold_set"]) and meta["gold_set"].issubset(ids)
            ),
            "candidate_target_slide": int(meta["target"] in ids),
        })

# Also record the ceiling of the FULL eligible set (the position-constrained
# universe): this is the absolute upper bound any reranker could reach.
for tid, meta in task_meta.items():
    ceiling_rows.append({
        "task_id": tid,
        "task_type": meta["task_type"],
        "candidate_source": "eligible_set_full",
        "candidate_count": meta["eligible_count"],
        "candidate_any_gold": meta["eligible_contains_any_gold"],
        "candidate_all_gold": meta["eligible_contains_all_gold"],
        "candidate_target_slide": meta["eligible_contains_target"],
    })

ceiling_detail = pd.DataFrame(ceiling_rows)

ceiling_detail.to_csv(
    OUT / "advanced_retrieval_candidate_ceiling_detail.csv",
    index=False,
)

pair_keys = sorted({
    (ti, idx)
    for (ti, _), cand in candidate_sets.items()
    for idx in cand
})

log(f"\nUnique query-document pairs to cross-encode: {len(pair_keys)}")


# ============================================================================
# Cross-encoder scoring (CPU)
# ============================================================================

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
)

log("Loading reranker:", RERANK_MODEL)

rerank_tokenizer = AutoTokenizer.from_pretrained(RERANK_MODEL)
reranker = AutoModelForSequenceClassification.from_pretrained(RERANK_MODEL)

reranker.to("cpu")
reranker.eval()

if torch.cuda.is_available():
    raise SystemExit("SAFETY STOP: CUDA visible after reranker load.")

ce_scores = {}

n_batches = math.ceil(len(pair_keys) / RERANK_BATCH)

log(f"Cross-encoder scoring: {n_batches} batches of {RERANK_BATCH} (CPU).")

t0 = time.time()

with torch.inference_mode():

    for batch_no, start_idx in enumerate(
        range(0, len(pair_keys), RERANK_BATCH),
        start=1,
    ):

        batch_keys = pair_keys[start_idx:start_idx + RERANK_BATCH]

        qbatch = [questions[ti] for ti, _ in batch_keys]
        dbatch = [docs_dense[idx] for _, idx in batch_keys]

        tok = rerank_tokenizer(
            qbatch,
            dbatch,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )

        out = reranker(**tok)

        logits = out.logits.view(-1).float().cpu().numpy()

        for key, score in zip(batch_keys, logits):
            ce_scores[key] = float(score)

        if batch_no == 1 or batch_no % 50 == 0 or batch_no == n_batches:

            done = start_idx + len(batch_keys)
            elapsed = time.time() - t0
            rate = done / max(elapsed, 1e-9)
            eta = (len(pair_keys) - done) / max(rate, 1e-9)

            log(
                f"  reranker batch {batch_no}/{n_batches} | "
                f"pairs {done}/{len(pair_keys)} | "
                f"elapsed={elapsed / 60:.1f} min | ETA={eta / 60:.1f} min"
            )

log(f"Cross-encoder scoring done in {(time.time() - t0) / 60:.1f} min")


rerank_rankings = {"bge50": {}, "union50": {}}
rerank_pool_size = {"bge50": {}, "union50": {}}

for (ti, cname), cand in candidate_sets.items():

    tid = str(tasks[ti]["task_id"])

    ranking_idx = sorted(
        cand,
        key=lambda idx: (-ce_scores[(ti, idx)], idx),
    )

    rerank_rankings[cname][tid] = [doc_ids[i] for i in ranking_idx]
    rerank_pool_size[cname][tid] = len(cand)

for cname in ["bge50", "union50"]:
    register(f"bge_reranker_{cname}", rerank_rankings[cname])

log("Reranked rankings built.")


# ============================================================================
# Aggregate and write outputs
# ============================================================================

ksweep_detail = pd.DataFrame(all_ksweep)
mrr_detail = pd.DataFrame(all_mrr)

ksweep_detail.to_csv(
    OUT / "advanced_retrieval_k_sweep_detail.csv",
    index=False,
)

mrr_detail.to_csv(
    OUT / "advanced_retrieval_mrr_detail.csv",
    index=False,
)

METHOD_ORDER = [
    "bm25",
    "rrf",
    "bge_dense",
    "hier_bge_L1",
    "hier_bge_L3",
    "hier_bge_L5",
    "bge_reranker_bge50",
    "bge_reranker_union50",
]


def order_methods(df):
    df = df.copy()
    df["__o"] = df["method"].map(
        {m: i for i, m in enumerate(METHOD_ORDER)}
    )
    df = df.sort_values(
        ["__o"] + [c for c in ["k", "task_scope"] if c in df.columns]
    )
    return df.drop(columns="__o")


def agg_ksweep(df, scope):
    out = (
        df.groupby(["method", "k"])
        .agg(
            N=("task_id", "size"),
            AnyGoldRecall=("any_gold", "mean"),
            AllGoldRecall=("all_gold", "mean"),
            TargetSlideRecall=("target_slide", "mean"),
            TargetLectureRecall=("target_lecture", "mean"),
            MeanRankingLength=("ranking_length", "mean"),
        )
        .reset_index()
    )
    out.insert(0, "task_scope", scope)
    return out


k_summary = pd.concat(
    [agg_ksweep(ksweep_detail, "ALL_ANSWERABLE")]
    + [
        agg_ksweep(g, tt)
        for tt, g in ksweep_detail.groupby("task_type")
    ],
    ignore_index=True,
)

k_summary = order_methods(k_summary)

k_summary.to_csv(
    OUT / "advanced_retrieval_k_sweep.csv",
    index=False,
)


def agg_mrr(df, scope):
    out = (
        df.groupby("method")
        .agg(
            N=("task_id", "size"),
            MRRAnyGold=("rr_any_gold", "mean"),
            MRRTargetSlide=("rr_target", "mean"),
            MedianFirstGoldRank=("first_gold_rank", "median"),
            MedianFirstTargetRank=("first_target_rank", "median"),
            FracGoldFoundInRanking=("gold_found", "mean"),
            FracTargetFoundInRanking=("target_found", "mean"),
            MeanRankingLength=("ranking_length", "mean"),
        )
        .reset_index()
    )
    out.insert(0, "task_scope", scope)
    return out


mrr_summary = pd.concat(
    [agg_mrr(mrr_detail, "ALL_ANSWERABLE")]
    + [
        agg_mrr(g, tt)
        for tt, g in mrr_detail.groupby("task_type")
    ],
    ignore_index=True,
)

mrr_summary = order_methods(mrr_summary)

mrr_summary.to_csv(
    OUT / "advanced_retrieval_mrr.csv",
    index=False,
)


ceiling_summary = pd.concat(
    [
        ceiling_detail.groupby("candidate_source")
        .agg(
            N=("task_id", "size"),
            MeanCandidateCount=("candidate_count", "mean"),
            MedianCandidateCount=("candidate_count", "median"),
            AnyGoldCeiling=("candidate_any_gold", "mean"),
            AllGoldCeiling=("candidate_all_gold", "mean"),
            TargetSlideCeiling=("candidate_target_slide", "mean"),
        )
        .reset_index()
        .assign(task_scope="ALL_ANSWERABLE"),

        ceiling_detail.groupby(["candidate_source", "task_type"])
        .agg(
            N=("task_id", "size"),
            MeanCandidateCount=("candidate_count", "mean"),
            MedianCandidateCount=("candidate_count", "median"),
            AnyGoldCeiling=("candidate_any_gold", "mean"),
            AllGoldCeiling=("candidate_all_gold", "mean"),
            TargetSlideCeiling=("candidate_target_slide", "mean"),
        )
        .reset_index()
        .rename(columns={"task_type": "task_scope"}),
    ],
    ignore_index=True,
)

ceiling_summary = ceiling_summary[
    [
        "task_scope",
        "candidate_source",
        "N",
        "MeanCandidateCount",
        "MedianCandidateCount",
        "AnyGoldCeiling",
        "AllGoldCeiling",
        "TargetSlideCeiling",
    ]
]

ceiling_summary.to_csv(
    OUT / "advanced_retrieval_candidate_ceiling.csv",
    index=False,
)


metadata = {
    "protocol": "position-constrained retrieval",
    "corpus": str(CORPUS),
    "corpus_docs": len(corpus),
    "benchmark": str(BENCH),
    "benchmark_tasks": len(all_tasks),
    "answerable_tasks": len(tasks),
    "task_identity_match_with_reference_protocol": TASK_IDENTITY_MATCH,
    "gate": gate_record,
    "embedding_model": EMBED_MODEL,
    "reranker_model": RERANK_MODEL,
    "dense_document_representation": "raw slide text",
    "lexical_document_representation": "reference doc_text()",
    "component_top_k": COMPONENT_TOP_K,
    "first_stage_k": FIRST_STAGE_K,
    "hierarchical_lecture_k": LECTURE_K,
    "k_sweep": K_SWEEP,
    "cross_encoder_pairs": len(pair_keys),
    "cpu_threads": CPU_THREADS,
    "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    "torch_cuda_available": bool(torch.cuda.is_available()),
    "python": platform.python_version(),
    "generation_performed": False,
}

(OUT / "advanced_retrieval_metadata.json").write_text(
    json.dumps(metadata, indent=2),
    encoding="utf-8",
)


# ============================================================================
# Console summary
# ============================================================================

overall_k = k_summary.query("task_scope == 'ALL_ANSWERABLE'")

log("")
log("=" * 100)
log("K-SWEEP (600 answerable tasks, position-constrained eligible sets)")
log("=" * 100)
log(
    overall_k[
        [
            "method",
            "k",
            "N",
            "AnyGoldRecall",
            "AllGoldRecall",
            "TargetSlideRecall",
            "TargetLectureRecall",
        ]
    ].to_string(index=False)
)

log("")
log("=" * 100)
log("MRR / MEDIAN RANKS")
log("=" * 100)
log(
    mrr_summary.query("task_scope == 'ALL_ANSWERABLE'")
    .to_string(index=False)
)

log("")
log("=" * 100)
log("CANDIDATE CEILINGS")
log("=" * 100)
log(
    ceiling_summary.query("task_scope == 'ALL_ANSWERABLE'")
    .to_string(index=False)
)

log("")
log("=" * 100)
log("DONE - retrieval only, no generation, CPU only")
log("=" * 100)
