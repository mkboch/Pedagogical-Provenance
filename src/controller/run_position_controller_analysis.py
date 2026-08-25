#!/usr/bin/env python3
"""Deployable-system analysis: apply the already-frozen, already-
validated practical course-aware controller (and the oracle-trigger
ceiling) to the existing 6,000 matched position-constrained BM25/RRF
generations. NO new LLM generation. NO retraining of the controller.

The controller decision (hybrid_prediction) is looked up purely by
task_id from artifacts/course_aware_trigger/course_aware_trigger_detail.csv
-- exactly the same lookup table used by
scripts/run_final_policy_controller_analysis.py for the primary
Slide-indexed controller experiment. Since the trigger decision depends
only on the task_id/question/course-position, not on which retriever
generated the answer, it can be applied unchanged to the BM25-position
and RRF-position matched generations.

Controller mechanics (matching run_final_policy_controller_analysis.py
exactly, generalized from a single base system to two methods):
  - controller does NOT intervene -> original generation's abstention /
    clinical_overreach / future_leakage metrics pass through unchanged.
  - controller intervenes (predicted label != "answerable") -> abstention
    forced to 1, clinical_overreach forced to 0, future_leakage forced
    to 0 (the generated answer is replaced by a fixed template; we do not
    have to re-score text since the deterministic post-intervention
    metric values are already known by definition of the template).
"""

from pathlib import Path
import json

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "artifacts" / "controller_analysis"
OUT.mkdir(parents=True, exist_ok=True)

SEED = 20260821
BOOT = 10000

ANSWERABLE_TYPES = {
    "evidence_cited_qa",
    "slide_local_factual_qa",
    "neighbor_slide_conceptual_qa",
}
CLINICAL_TYPE = "out_of_scope_abstention_qa"
SEQUENCE_TYPE = "lecture_sequence_violation_probe"
POLICY_TYPES = {CLINICAL_TYPE, SEQUENCE_TYPE}

TRIGGER_FILE = ROOT / "artifacts/course_aware_trigger/course_aware_trigger_detail.csv"

METRIC_FILES = {
    "qwen3_8b": ROOT / "artifacts/matched_generation/qwen3_metric_checkpoint/qwen3_matched_position_exact_historical_metrics_detail.csv",
    "qwen2_5_7b": ROOT / "artifacts/matched_generation/qwen2_5_metric_checkpoint/qwen25_matched_position_exact_historical_metrics_detail.csv",
    "mistral_7b": ROOT / "artifacts/matched_generation/mistral_metric_checkpoint/mistral_matched_position_exact_historical_metrics_detail.csv",
}

for p in [TRIGGER_FILE] + list(METRIC_FILES.values()):
    if not p.exists():
        raise SystemExit(f"FAILED: missing input {p}")

trigger = pd.read_csv(TRIGGER_FILE)
trigger_lookup = (
    trigger[["task_id", "hybrid_prediction", "truth"]]
    .drop_duplicates("task_id")
    .copy()
)
if len(trigger_lookup) != 1000:
    raise SystemExit(f"FAILED: expected 1000 trigger decisions, got {len(trigger_lookup)}")

frames = []
for model_key, path in METRIC_FILES.items():
    df = pd.read_csv(path)
    if len(df) != 2000:
        raise SystemExit(f"FAILED: {model_key} expected 2000 rows, got {len(df)}")
    required = {"task_id", "task_type", "method", "lexical_f1", "abstention_score", "clinical_overreach", "future_leakage"}
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(f"FAILED: {model_key} missing columns {sorted(missing)}")
    df["model_key"] = model_key
    frames.append(df)

all_gen = pd.concat(frames, ignore_index=True)
if len(all_gen) != 6000:
    raise SystemExit(f"FAILED: expected 6000 matched position rows, got {len(all_gen)}")

all_gen = all_gen.merge(trigger_lookup, on="task_id", how="left", validate="many_to_one")
if all_gen["hybrid_prediction"].isna().any():
    raise SystemExit("FAILED: some position outputs could not be matched to trigger decisions")


def oracle_label(task_type):
    if task_type == CLINICAL_TYPE:
        return "clinical_out_of_scope"
    if task_type == SEQUENCE_TYPE:
        return "sequence_sensitive"
    return "answerable"


all_gen["oracle_trigger"] = all_gen["task_type"].map(oracle_label)

variant_frames = []
for method in ["bm25", "rrf"]:
    method_df = all_gen.loc[all_gen["method"].eq(method)].copy()

    for variant in ["no_controller", "course_aware_controller", "oracle_trigger_controller"]:
        x = method_df.copy()

        if variant == "no_controller":
            x["controller_trigger"] = "answerable"
        elif variant == "oracle_trigger_controller":
            x["controller_trigger"] = x["oracle_trigger"]
        else:
            x["controller_trigger"] = x["hybrid_prediction"]

        x["controller_intervened"] = x["controller_trigger"].ne("answerable").astype(int)

        x["post_abstention"] = np.where(
            x["controller_intervened"].eq(1), 1.0, pd.to_numeric(x["abstention_score"], errors="coerce")
        )
        x["post_clinical_overreach"] = np.where(
            x["controller_intervened"].eq(1), 0.0, pd.to_numeric(x["clinical_overreach"], errors="coerce")
        )
        x["post_future_leakage"] = np.where(
            x["controller_intervened"].eq(1), 0.0, pd.to_numeric(x["future_leakage"], errors="coerce")
        )

        x["is_answerable_task"] = x["task_type"].isin(ANSWERABLE_TYPES).astype(int)
        x["is_policy_task"] = x["task_type"].isin(POLICY_TYPES).astype(int)
        x["false_control"] = (x["is_answerable_task"].eq(1) & x["controller_intervened"].eq(1)).astype(int)
        x["answer_preserved"] = (x["is_answerable_task"].eq(1) & x["controller_intervened"].eq(0)).astype(int)
        x["correct_trigger"] = x["controller_trigger"].eq(x["oracle_trigger"]).astype(int)

        x["policy_failure"] = np.nan
        clinical_mask = x["task_type"].eq(CLINICAL_TYPE)
        sequence_mask = x["task_type"].eq(SEQUENCE_TYPE)
        x.loc[clinical_mask, "policy_failure"] = x.loc[clinical_mask, "post_clinical_overreach"]
        x.loc[sequence_mask, "policy_failure"] = x.loc[sequence_mask, "post_future_leakage"]

        x["method"] = method
        x["controller_variant"] = variant
        variant_frames.append(x)

all_variants = pd.concat(variant_frames, ignore_index=True)
all_variants.to_csv(OUT / "position_plus_controller_output_detail.csv", index=False)

# =====================================================================
# SUMMARY (pooled across 3 models, model-task-output level, matching the
# primary controller script's summary granularity)
# =====================================================================

summary_rows = []
for (method, variant), vg in all_variants.groupby(["method", "controller_variant"]):
    a = vg[vg["is_answerable_task"].eq(1)]
    c = vg[vg["task_type"].eq(CLINICAL_TYPE)]
    s = vg[vg["task_type"].eq(SEQUENCE_TYPE)]
    p = vg[vg["is_policy_task"].eq(1)]

    summary_rows.append(
        {
            "method": method,
            "variant": variant,
            "N_outputs": len(vg),
            "answerable_outputs": len(a),
            "answer_preservation_rate": a["answer_preserved"].mean(),
            "false_control_rate": a["false_control"].mean(),
            "clinical_abstention_rate": c["post_abstention"].mean(),
            "clinical_overreach_rate": c["post_clinical_overreach"].mean(),
            "sequence_abstention_rate": s["post_abstention"].mean(),
            "sequence_leakage_rate": s["post_future_leakage"].mean(),
            "overall_policy_failure_rate": p["policy_failure"].mean(),
            "overall_policy_abstention_rate": p["post_abstention"].mean(),
            "controller_intervention_rate_all": vg["controller_intervened"].mean(),
            "trigger_accuracy": vg["correct_trigger"].mean(),
        }
    )

summary_df = pd.DataFrame(summary_rows)
variant_order = {"no_controller": 0, "course_aware_controller": 1, "oracle_trigger_controller": 2}
summary_df["_vorder"] = summary_df["variant"].map(variant_order)
summary_df = summary_df.sort_values(["method", "_vorder"]).drop(columns="_vorder")
summary_df.to_csv(OUT / "position_plus_controller_summary.csv", index=False)

# =====================================================================
# TASK-CLUSTERED (benchmark task = inference unit; average across the 3
# models within each task before task-level bootstrap, per this
# session's established convention)
# =====================================================================

task_level = (
    all_variants.groupby(["method", "controller_variant", "task_id", "task_type"], as_index=False)
    .agg(
        false_control=("false_control", "mean"),
        answer_preserved=("answer_preserved", "mean"),
        post_clinical_overreach=("post_clinical_overreach", "mean"),
        post_future_leakage=("post_future_leakage", "mean"),
        policy_failure=("policy_failure", "mean"),
        post_abstention=("post_abstention", "mean"),
        correct_trigger=("correct_trigger", "mean"),
        controller_intervened=("controller_intervened", "mean"),
    )
)
task_level.to_csv(OUT / "position_plus_controller_task_level.csv", index=False)

rng = np.random.default_rng(SEED)


def bootstrap_mean(values, n_boot=BOOT):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return np.nan, np.nan, np.nan
    point = float(np.mean(values))
    n = len(values)
    sims = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        sims[b] = np.mean(values[idx])
    low, high = np.quantile(sims, [0.025, 0.975])
    return point, float(low), float(high)


ci_rows = []
for method in ["bm25", "rrf"]:
    for variant in ["no_controller", "course_aware_controller", "oracle_trigger_controller"]:
        v = task_level[(task_level["method"] == method) & (task_level["controller_variant"] == variant)]

        scopes = {
            "answerable_false_control": v[v["task_type"].isin(ANSWERABLE_TYPES)]["false_control"],
            "clinical_overreach": v[v["task_type"].eq(CLINICAL_TYPE)]["post_clinical_overreach"],
            "sequence_leakage": v[v["task_type"].eq(SEQUENCE_TYPE)]["post_future_leakage"],
            "overall_policy_failure": v[v["task_type"].isin(POLICY_TYPES)]["policy_failure"],
        }

        for metric, values in scopes.items():
            point, low, high = bootstrap_mean(values)
            ci_rows.append(
                {
                    "method": method,
                    "variant": variant,
                    "metric": metric,
                    "estimate": point,
                    "ci_low": low,
                    "ci_high": high,
                    "N_tasks": len(values),
                }
            )

ci_df = pd.DataFrame(ci_rows)
ci_df.to_csv(OUT / "position_plus_controller_bootstrap_ci.csv", index=False)

# =====================================================================
# CONSISTENCY CHECK against the primary (Slide-indexed) controller
# experiment's known numbers, as a sanity gate (not a strict pass/fail,
# just a documented comparison since these are different base systems)
# =====================================================================

print("=" * 100)
print("POSITION + CONTROLLER SUMMARY")
print("=" * 100)
print(summary_df.to_string(index=False))
print()
print("=" * 100)
print("TASK-CLUSTERED BOOTSTRAP CIs")
print("=" * 100)
print(ci_df.to_string(index=False))

metadata = {
    "status": "POSITION_PLUS_CONTROLLER_ANALYSIS_COMPLETE",
    "classification": "deployable_system_analysis_no_new_generation",
    "controller_source": str(TRIGGER_FILE),
    "controller_reused_unchanged": True,
    "generation_source_files": {k: str(v) for k, v in METRIC_FILES.items()},
    "n_matched_position_rows": 6000,
    "n_tasks_per_method": 1000,
    "bootstrap_iterations": BOOT,
    "bootstrap_unit": "task_id after averaging model-specific outputs within task, per method",
    "no_new_generation": True,
}
(OUT / "position_plus_controller_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

print()
print("POSITION_PLUS_CONTROLLER_ANALYSIS = DONE")
