#!/usr/bin/env python3
from pathlib import Path
import pandas as pd
import json
import sys

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "artifacts/final_position_constrained_analysis"
OUT.mkdir(parents=True, exist_ok=True)

files = {
    "qwen3_8b": ROOT / "artifacts/position_constrained_generation_qwen3_8b/qwen3_8b_position_constrained_generation_metrics.csv",
    "qwen25_7b_instruct": ROOT / "artifacts/position_constrained_generation_qwen25_7b/qwen25_7b_instruct_position_constrained_generation_metrics.csv",
    "mistral_7b_instruct_v03": ROOT / "artifacts/position_constrained_generation_mistral_7b/mistral_7b_instruct_v03_position_constrained_generation_metrics.csv",
}

print("===== CHECK INPUT FILES =====")
missing = []
for name, p in files.items():
    print(name, "=>", p, "EXISTS" if p.exists() else "MISSING")
    if not p.exists():
        missing.append(str(p))

if missing:
    print("\nFAILED: missing required model output files:")
    for p in missing:
        print(" -", p)
    sys.exit(1)

dfs = []
for name, p in files.items():
    df = pd.read_csv(p)
    df["model_name"] = name
    dfs.append(df)

pos = pd.concat(dfs, ignore_index=True)

expected = 3 * 1000 * 3
print("\nposition_constrained_rows:", len(pos), "expected:", expected)
if len(pos) != expected:
    print("WARNING: row count is not expected 9000. Continuing but inspect carefully.")

metrics = {
    "F1": "lexical_f1_vs_reference",
    "Evidence": "evidence_token_overlap",
    "Gold": "cites_gold_evidence",
    "Fake": "fake_slide_citation",
    "Abst": "abstention_score",
    "ClinOver": "clinical_overreach",
    "Leak": "future_leakage",
}

answerable_tasks = {
    "evidence_cited_qa",
    "slide_local_factual_qa",
    "neighbor_slide_conceptual_qa",
}
policy_tasks = {
    "out_of_scope_abstention_qa",
    "lecture_sequence_violation_probe",
}

def summarize(df, group_cols):
    rows = []
    for keys, g in df.groupby(group_cols):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_cols, keys))
        row["N"] = len(g)
        for k, c in metrics.items():
            row[k] = g[c].mean()
        rows.append(row)
    return pd.DataFrame(rows)

# Position-constrained summaries.
pos_all = summarize(pos, ["system_name"])
pos_all.to_csv(OUT / "position_constrained_generation_3model_summary_all_tasks.csv", index=False)

pos_answerable = summarize(pos[pos["task_type"].isin(answerable_tasks)], ["system_name"])
pos_answerable.to_csv(OUT / "position_constrained_generation_3model_summary_answerable_tasks.csv", index=False)

pos_policy = summarize(pos[pos["task_type"].isin(policy_tasks)], ["system_name"])
pos_policy.to_csv(OUT / "position_constrained_generation_3model_summary_policy_tasks.csv", index=False)

pos_by_task = summarize(pos, ["task_type", "system_name"])
pos_by_task.to_csv(OUT / "position_constrained_generation_3model_summary_by_task.csv", index=False)

pos_by_model = summarize(pos, ["model_name", "system_name"])
pos_by_model.to_csv(OUT / "position_constrained_generation_by_model.csv", index=False)

# Original full detail.
orig_candidates = [
    ROOT / "artifacts/paper_tables/full1000_final_statistics/full1000_all_models_metrics_detail.csv",
    ROOT / "artifacts/core_evaluation/full_task_metrics_detail.csv",
]
orig_path = next((p for p in orig_candidates if p.exists()), None)
if orig_path is None:
    matches = sorted(ROOT.glob("**/full1000_all_models_metrics_detail.csv"))
    orig_path = matches[0] if matches else None

if orig_path is None:
    print("WARNING: original full1000 detail not found; only position-constrained tables written.")
else:
    orig = pd.read_csv(orig_path)
    print("\noriginal_detail:", orig_path, "rows:", len(orig))

    # Harmonize columns.
    keep_cols = [
        "model_name", "system_name", "task_id", "task_type",
        "lexical_f1_vs_reference", "evidence_token_overlap",
        "cites_gold_evidence", "fake_slide_citation",
        "abstention_score", "clinical_overreach", "future_leakage"
    ]
    orig_small = orig[keep_cols].copy()
    pos_small = pos[keep_cols].copy()

    combined = pd.concat([orig_small, pos_small], ignore_index=True)
    combined.to_csv(OUT / "combined_original_plus_position_constrained_metrics.csv", index=False)

    display_map = {
        "vanilla_llm": "Vanilla LLM",
        "standard_bm25_rag": "BM25 global RAG",
        "standard_rrf_bm25_tfidf_rag": "RRF global RAG",
        "slide_indexed_pedagogical_rag": "Oracle slide-indexed RAG",
        "slide_indexed_pedagogical_rag_plus_verifier": "Oracle slide-indexed RAG + verifier",
        "bm25_position_constrained_rag": "BM25 position-constrained RAG",
        "rrf_position_constrained_rag": "RRF position-constrained RAG",
        "bge_position_constrained_rag": "BGE position-constrained RAG",
    }
    order = [
        "Vanilla LLM",
        "BM25 global RAG",
        "BM25 position-constrained RAG",
        "RRF global RAG",
        "RRF position-constrained RAG",
        "BGE position-constrained RAG",
        "Oracle slide-indexed RAG",
        "Oracle slide-indexed RAG + verifier",
    ]

    def paper_summary(df, name):
        t = summarize(df, ["system_name"])
        t["System"] = t["system_name"].map(display_map).fillna(t["system_name"])
        t["_order"] = t["System"].map({s:i for i,s in enumerate(order)}).fillna(999)
        t = t.sort_values("_order")
        t = t[["System", "N", "F1", "Evidence", "Gold", "Fake", "Abst", "ClinOver", "Leak"]]
        t.to_csv(OUT / name, index=False)
        return t

    all_table = paper_summary(combined, "paper_table_combined_all_tasks.csv")
    answerable_table = paper_summary(combined[combined["task_type"].isin(answerable_tasks)], "paper_table_combined_answerable_tasks.csv")
    policy_table = paper_summary(combined[combined["task_type"].isin(policy_tasks)], "paper_table_combined_policy_tasks.csv")

    task_table = summarize(combined, ["task_type", "system_name"])
    task_table["System"] = task_table["system_name"].map(display_map).fillna(task_table["system_name"])
    task_table["_order"] = task_table["System"].map({s:i for i,s in enumerate(order)}).fillna(999)
    task_table = task_table.sort_values(["task_type", "_order"])
    task_table = task_table[["task_type", "System", "N", "F1", "Evidence", "Gold", "Fake", "Abst", "ClinOver", "Leak"]]
    task_table.to_csv(OUT / "paper_table_combined_by_task.csv", index=False)

    print("\n===== PAPER TABLE: ANSWERABLE TASKS =====")
    print(answerable_table.to_string(index=False))

    print("\n===== PAPER TABLE: POLICY-CONTROL TASKS =====")
    print(policy_table.to_string(index=False))

    print("\n===== PAPER TABLE: ALL TASKS DIAGNOSTIC =====")
    print(all_table.to_string(index=False))

print("\n===== POSITION-CONSTRAINED BY MODEL =====")
print(pos_by_model.to_string(index=False))

meta = {
    "position_files": {k: str(v) for k, v in files.items()},
    "rows_position_constrained": int(len(pos)),
    "outputs_dir": str(OUT),
}
(OUT / "merge_metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

print("\n===== OUTPUTS =====")
for p in sorted(OUT.glob("*")):
    print(p)

print("\nMERGE POSITION-CONSTRAINED GENERATION OK")
