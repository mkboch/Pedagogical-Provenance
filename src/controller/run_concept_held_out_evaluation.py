#!/usr/bin/env python3
"""analysis stage/5: Concept-held-out evaluation of the EXISTING practical
course-aware trigger, on Canonical Multi-Concept Sequence Set.

Key scientific question this answers, which Core QA Benchmark CANNOT:
  Core QA Benchmark asks "can the trigger recognize the frozen sequence-policy
  task distribution?" (2 concepts, concept == f(lecture)).
  Canonical Multi-Concept Sequence Set asks "does the trigger generalize to HELD-OUT COURSE
  CONCEPTS it never saw in training?"

Fidelity requirements honoured here:
  - The controller's feature definitions, concept extraction, embedding
    model, and classifier hyperparameters are reused EXACTLY as defined in
    src/controller/run_course_aware_controller.py (AST-recovered, not reimplemented).
  - Grouping is by CANONICAL CONCEPT (not lecture): the same concept can
    never appear in both a train and a validation fold.
  - No hyperparameter is tuned against evaluation labels.
  - Retrieval/embedding encoder use only. NO LLM answer generation.
"""

from pathlib import Path
import ast
import json
import re

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (confusion_matrix, precision_recall_fscore_support,
                             roc_auc_score, average_precision_score,
                             balanced_accuracy_score, accuracy_score, f1_score)

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "artifacts" / "sequence_policy"
SRC = ROOT / "scripts" / "run_course_aware_trigger.py"
CORPUS = ROOT / "data/corpus/slide_corpus_final.jsonl"
EXT = ROOT / "data" / "benchmark" / "canonical_multiconcept_sequence.jsonl"

SEED = 20260817          # identical to the original controller
EMBED_MODEL = "BAAI/bge-base-en-v1.5"
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

FEATURES = [
    "lex_log_past_docs", "lex_log_future_docs", "lex_future_fraction",
    "sem_past_max", "sem_future_max", "sem_delta_future_past",
    "sem_past_top3", "sem_future_top3", "sem_delta_top3",
    "sem_top10_future_fraction", "sem_first_available_rank",
    "position_fraction",
]


def recover(names):
    """AST-recover helpers verbatim from the original controller.

    Recovers EVERY top-level function definition (not a hand-picked subset)
    so that transitive dependencies among the controller's own helpers
    (e.g. extract_concept -> clean_spaces, phrase_pattern -> normalize_text)
    are satisfied by construction rather than discovered one failure at a
    time. `names` is then asserted to be present, so the fidelity guarantee
    is unchanged: the functions used are byte-identical to the originals.
    """
    tree = ast.parse(SRC.read_text(encoding="utf-8"))
    body = [n for n in tree.body if isinstance(n, ast.FunctionDef)]
    consts = [n for n in tree.body if isinstance(n, (ast.Assign, ast.AnnAssign))]
    keep = []
    for n in consts:
        try:
            ast.literal_eval(n.value)
            keep.append(n)
        except Exception:
            pass
    mod = ast.Module(body=keep + body, type_ignores=[])
    ast.fix_missing_locations(mod)
    ns = {"re": re, "np": np, "json": json, "math": __import__("math")}
    exec(compile(mod, str(SRC), "exec"), ns, ns)
    missing = [n for n in names if n not in ns]
    if missing:
        raise SystemExit(f"FAILED to recover: {missing}")
    return ns


ns = recover(["clean_spaces", "strip_cues", "extract_concept", "clinical_intent",
              "phrase_pattern", "top_mean"])
strip_cues = ns["strip_cues"]
extract_concept = ns["extract_concept"]
clinical_intent = ns["clinical_intent"]
phrase_pattern = ns["phrase_pattern"]
top_mean = ns["top_mean"]
print("Recovered controller helpers verbatim:",
      ["clean_spaces", "strip_cues", "extract_concept", "clinical_intent", "phrase_pattern", "top_mean"])


def norm(s):
    return re.sub(r"\s+", " ", str(s or "").lower()).strip()


# ---------------------------------------------------------------------
# LOAD
# ---------------------------------------------------------------------
corpus = [json.loads(l) for l in CORPUS.read_text(encoding="utf-8").splitlines() if l.strip()]
for d in corpus:
    d["lecture_id"] = int(d["lecture_id"]); d["slide_id"] = int(d["slide_id"])
    d["_norm"] = norm(d.get("text", ""))
ordered = sorted(corpus, key=lambda d: (d["lecture_id"], d["slide_id"]))
for i, d in enumerate(ordered):
    d["course_position"] = i
n_lectures = len({d["lecture_id"] for d in ordered})
max_lecture = max(d["lecture_id"] for d in ordered)

tasks = [json.loads(l) for l in EXT.read_text(encoding="utf-8").splitlines() if l.strip()]
print(f"Sequence tasks: {len(tasks)}   corpus: {len(ordered)}")

# ---------------------------------------------------------------------
# FEATURES  (identical definitions to run_course_aware_trigger.py)
# ---------------------------------------------------------------------
from sentence_transformers import SentenceTransformer
import torch
if torch.cuda.is_available():
    print("NOTE: CUDA visible; forcing CPU to match original controller protocol.")
model = SentenceTransformer(EMBED_MODEL, device="cpu")
print("max_seq_length =", model.max_seq_length)

doc_texts = [d.get("text", "") for d in ordered]
doc_emb = model.encode(doc_texts, batch_size=32, convert_to_numpy=True,
                       normalize_embeddings=True, show_progress_bar=True)

rows = []
for t in tasks:
    q = t["question"]
    md = t["metadata"]
    stripped = strip_cues(q)
    concept = extract_concept(stripped)          # extracted from TEXT, as in deployment
    anchor_pos = int(md["anchor_course_position"])
    lecture = int(md["anchor_lecture"])

    pat = phrase_pattern(concept) if concept else None
    lex_past_docs = lex_future_docs = 0
    if pat is not None:
        for d in ordered:
            if pat.search(d["_norm"]):
                if d["course_position"] <= anchor_pos:
                    lex_past_docs += 1
                else:
                    lex_future_docs += 1
    lex_total = lex_past_docs + lex_future_docs

    cemb = model.encode([QUERY_PREFIX + (concept or q)], convert_to_numpy=True,
                        normalize_embeddings=True)[0]
    sims = doc_emb @ cemb
    past_mask = np.array([d["course_position"] <= anchor_pos for d in ordered])
    past_s, fut_s = sims[past_mask], sims[~past_mask]
    order = np.argsort(-sims)
    first_avail = next((r + 1 for r, i in enumerate(order) if past_mask[i]), len(sims))
    top10 = order[:10]

    rows.append({
        "task_id": t["task_id"],
        "concept_canonical": md["concept"],
        "concept_extracted": concept,
        "sequence_label": md["sequence_label"],
        "policy_class": md["policy_class"],
        "anchor_lecture": lecture,
        "anchor_course_position": anchor_pos,
        "clinical_rule": int(clinical_intent(stripped)),
        "lex_past_docs": lex_past_docs,
        "lex_future_docs": lex_future_docs,
        "lex_log_past_docs": float(np.log1p(lex_past_docs)),
        "lex_log_future_docs": float(np.log1p(lex_future_docs)),
        "lex_future_fraction": float(lex_future_docs / lex_total) if lex_total else 0.0,
        "sem_past_max": float(past_s.max()) if past_s.size else 0.0,
        "sem_future_max": float(fut_s.max()) if fut_s.size else 0.0,
        "sem_delta_future_past": float((fut_s.max() if fut_s.size else 0.0) - (past_s.max() if past_s.size else 0.0)),
        "sem_past_top3": float(top_mean(past_s)) if past_s.size else 0.0,
        "sem_future_top3": float(top_mean(fut_s)) if fut_s.size else 0.0,
        "sem_delta_top3": float((top_mean(fut_s) if fut_s.size else 0.0) - (top_mean(past_s) if past_s.size else 0.0)),
        "sem_top10_future_fraction": float(np.mean([not past_mask[i] for i in top10])),
        "sem_first_available_rank": float(first_avail),
        "position_fraction": float(lecture / max(max_lecture, 1)),
    })

feat = pd.DataFrame(rows)
feat["y"] = (feat["sequence_label"] == "not_yet_available").astype(int)
feat.to_csv(OUT / "concept_grouped_controller_features.csv", index=False)

print("\nConcept extraction fidelity (canonical vs extracted-from-text):")
match = (feat["concept_canonical"].str.lower() == feat["concept_extracted"].str.lower()).mean()
print(f"  exact match rate: {match:.3f}")

# ---------------------------------------------------------------------
# CONCEPT-GROUPED CROSS-VALIDATION
# ---------------------------------------------------------------------
X = feat[FEATURES].to_numpy(float)
y = feat["y"].to_numpy(int)
groups = feat["concept_canonical"].to_numpy()
n_concepts = len(set(groups))
n_splits = 5
print(f"\nConcepts: {n_concepts}   folds: {n_splits}   grouping=CANONICAL CONCEPT")

sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
oof_prob = np.full(len(y), np.nan)
fold_rows = []
for k, (tr, te) in enumerate(sgkf.split(X, y, groups), 1):
    pipe = Pipeline([("sc", StandardScaler()),
                     ("lr", LogisticRegression(C=1.0, class_weight="balanced",
                                               max_iter=3000, random_state=SEED))])
    pipe.fit(X[tr], y[tr])
    p = pipe.predict_proba(X[te])[:, 1]
    oof_prob[te] = p
    pred = (p >= 0.5).astype(int)
    tr_c, te_c = set(groups[tr]), set(groups[te])
    assert not (tr_c & te_c), "CONCEPT LEAKAGE ACROSS FOLDS"
    fold_rows.append({
        "fold": k, "n_train": len(tr), "n_test": len(te),
        "n_train_concepts": len(tr_c), "n_test_concepts": len(te_c),
        "test_concepts": "|".join(sorted(te_c)),
        "test_pos_rate": float(y[te].mean()),
        "accuracy": accuracy_score(y[te], pred),
        "balanced_accuracy": balanced_accuracy_score(y[te], pred),
        "macro_f1": f1_score(y[te], pred, average="macro"),
    })

folds = pd.DataFrame(fold_rows)
folds.to_csv(OUT / "concept_grouped_controller_fold_results.csv", index=False)
print("\nPER-FOLD (concept-held-out):")
print(folds[["fold","n_train","n_test","n_train_concepts","n_test_concepts",
             "accuracy","balanced_accuracy","macro_f1"]].to_string(index=False))

feat["oof_prob"] = oof_prob
feat["oof_pred"] = (oof_prob >= 0.5).astype(int)
feat.to_csv(OUT / "concept_grouped_controller_predictions.csv", index=False)

pred = feat["oof_pred"].to_numpy()
cm = confusion_matrix(y, pred, labels=[0, 1])
pd.DataFrame(cm, index=["true_available", "true_not_yet_available"],
             columns=["pred_available", "pred_not_yet_available"]).to_csv(
    OUT / "concept_grouped_controller_confusion_matrix.csv")

pr, rc, f1, sup = precision_recall_fscore_support(y, pred, labels=[0, 1], zero_division=0)
metrics = {
    "n_tasks": int(len(y)), "n_concepts": int(n_concepts), "n_folds": n_splits,
    "grouping": "canonical_concept",
    "accuracy": float(accuracy_score(y, pred)),
    "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
    "macro_f1": float(f1_score(y, pred, average="macro")),
    "roc_auc": float(roc_auc_score(y, oof_prob)),
    "pr_auc_not_yet_available": float(average_precision_score(y, oof_prob)),
    "precision_available": float(pr[0]), "recall_available": float(rc[0]), "f1_available": float(f1[0]),
    "precision_not_yet_available": float(pr[1]), "recall_not_yet_available": float(rc[1]),
    "f1_not_yet_available": float(f1[1]),
    "sequence_recall": float(rc[1]),
    "false_control_rate_on_available": float(1.0 - rc[0]),
    "support_available": int(sup[0]), "support_not_yet_available": int(sup[1]),
}
pd.DataFrame([metrics]).to_csv(OUT / "concept_grouped_controller_metrics.csv", index=False)

print("\nCONFUSION MATRIX (rows=truth, cols=pred):")
print(pd.DataFrame(cm, index=["true_available","true_not_yet"],
                   columns=["pred_available","pred_not_yet"]).to_string())
print("\nHEADLINE (concept-held-out OOF):")
for k in ["accuracy","balanced_accuracy","macro_f1","roc_auc","pr_auc_not_yet_available",
          "sequence_recall","false_control_rate_on_available"]:
    print(f"  {k:34s} {metrics[k]:.4f}")

# ---------------------------------------------------------------------
# THRESHOLD / OPERATING-CURVE ANALYSIS
# ---------------------------------------------------------------------
curve = []
for thr in np.round(np.arange(0.05, 0.96, 0.05), 2):
    p = (oof_prob >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, p, labels=[0, 1]).ravel()
    curve.append({
        "threshold": float(thr),
        "sequence_recall": tp / (tp + fn) if (tp + fn) else np.nan,
        "false_control_rate_on_available": fp / (fp + tn) if (fp + tn) else np.nan,
        "answer_preservation_on_available": tn / (fp + tn) if (fp + tn) else np.nan,
        "trigger_precision": tp / (tp + fp) if (tp + fp) else np.nan,
        "intervention_rate": float(p.mean()),
        "accuracy": float(accuracy_score(y, p)),
        "balanced_accuracy": float(balanced_accuracy_score(y, p)),
        "macro_f1": float(f1_score(y, p, average="macro")),
    })
cdf = pd.DataFrame(curve)
cdf.to_csv(OUT / "controller_threshold_curve_sequence.csv", index=False)
print("\nOPERATING CURVE (concept-held-out):")
print(cdf[["threshold","sequence_recall","false_control_rate_on_available",
           "trigger_precision","balanced_accuracy"]].to_string(index=False))

print("\nCONCEPT_HELD_OUT_CONTROLLER_EVAL = DONE  (no LLM generation)")
