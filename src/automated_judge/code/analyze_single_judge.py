#!/usr/bin/env python3
"""Validate and analyze the two clean orientations for one automated judge."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest, pearsonr


SEED = 20260919
SCORES = ["correctness", "completeness_relevance", "evidence_faithfulness"]


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def parsed_frame(root: Path, judge_dir: Path, prefix: str, pass_name: str, mapping: pd.DataFrame) -> pd.DataFrame:
    raw = read_jsonl(judge_dir / f"{prefix}_{pass_name}_raw_600.jsonl")
    if len(raw) != 600 or len({row["task_id"] for row in raw}) != 600:
        raise RuntimeError(f"{prefix} {pass_name}: expected 600 unique raw records")
    map_index = mapping.set_index("task_id")
    expected_fields = {f"{score}_{position}" for score in SCORES for position in "AB"} | {"overall_preference", "brief_reason"}
    rows = []
    for record in raw:
        parsed = record.get("parsed")
        if not isinstance(parsed, dict) or set(parsed) != expected_fields:
            raise RuntimeError(f"{prefix} {pass_name}: parser gate failed for {record['task_id']}")
        if any(isinstance(parsed[f"{score}_{position}"], bool) or not isinstance(parsed[f"{score}_{position}"], int)
               or not 1 <= parsed[f"{score}_{position}"] <= 5 for score in SCORES for position in "AB"):
            raise RuntimeError(f"{prefix} {pass_name}: score gate failed for {record['task_id']}")
        if parsed["overall_preference"] not in {"A", "B", "TIE"} or len(parsed["brief_reason"].split()) > 50:
            raise RuntimeError(f"{prefix} {pass_name}: preference/reason gate failed for {record['task_id']}")
        metadata = map_index.loc[record["task_id"]]
        a_condition = metadata[f"{pass_name}_A_condition"]
        b_condition = metadata[f"{pass_name}_B_condition"]
        if parsed["overall_preference"] == "TIE":
            mapped_preference = "tie"
        else:
            preferred_condition = a_condition if parsed["overall_preference"] == "A" else b_condition
            mapped_preference = "full_pacer" if preferred_condition == "full_pacer_lambdamart" else "content_only"
        row = {
            "task_id": record["task_id"], "task_type": metadata["task_type"],
            "target_lecture": metadata["target_lecture"], "A_condition": a_condition,
            "B_condition": b_condition, **parsed, "mapped_preference": mapped_preference,
            "selected_attempt": record["selected_attempt"], "attempt_count": len(record["attempts"]),
        }
        for score in SCORES:
            row[f"content_only_{score}"] = parsed[f"{score}_A"] if a_condition == "content_only_lambdamart" else parsed[f"{score}_B"]
            row[f"full_pacer_{score}"] = parsed[f"{score}_A"] if a_condition == "full_pacer_lambdamart" else parsed[f"{score}_B"]
        rows.append(row)
    return pd.DataFrame(rows).sort_values("task_id").reset_index(drop=True)


def cluster_ci(frame: pd.DataFrame, diff_col: str) -> tuple[float, float, int, int, int]:
    grouped = frame.groupby("target_lecture")[diff_col].agg(["sum", "size", "mean"]).sort_index()
    sums = grouped["sum"].to_numpy(float)
    sizes = grouped["size"].to_numpy(float)
    rng = np.random.default_rng(SEED)
    replicates = np.empty(100_000)
    done = 0
    while done < len(replicates):
        block = min(5_000, len(replicates) - done)
        sample = rng.integers(0, len(grouped), size=(block, len(grouped)))
        replicates[done:done + block] = sums[sample].sum(axis=1) / sizes[sample].sum(axis=1)
        done += block
    low, high = np.quantile(replicates, [0.025, 0.975])
    lecture_means = grouped["mean"].to_numpy()
    return float(low), float(high), int((lecture_means > 0).sum()), int((lecture_means == 0).sum()), int((lecture_means < 0).sum())


def markdown_table(frame: pd.DataFrame) -> str:
    display = frame.copy()
    for column in display.select_dtypes(include=["float"]).columns:
        display[column] = display[column].map(lambda value: "" if pd.isna(value) else f"{value:.6f}")
    columns = list(display.columns)
    lines = ["| " + " | ".join(columns) + " |", "|" + "|".join(["---"] * len(columns)) + "|"]
    for row in display.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(str(value).replace("|", "\\|").replace("\n", " ") for value in row) + " |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--judge-subdir", required=True)
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--display-name", required=True)
    parser.add_argument("--report-filename", required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    judge_dir = root / args.judge_subdir
    mapping = pd.read_csv(root / "blinded_condition_mapping.csv")
    primary = parsed_frame(root, judge_dir, args.prefix, "primary", mapping)
    swapped = parsed_frame(root, judge_dir, args.prefix, "swapped", mapping)
    primary.to_csv(judge_dir / f"{args.prefix}_primary_parsed_600.csv", index=False)
    swapped.to_csv(judge_dir / f"{args.prefix}_swapped_parsed_600.csv", index=False)

    wins = int((primary.mapped_preference == "full_pacer").sum())
    losses = int((primary.mapped_preference == "content_only").sum())
    ties = int((primary.mapped_preference == "tie").sum())
    sign_p = float(binomtest(wins, wins + losses, 0.5).pvalue) if wins + losses else 1.0
    preference_rows = [
        {"section": "preference", "metric": "full_pacer_wins", "value": wins},
        {"section": "preference", "metric": "content_only_wins", "value": losses},
        {"section": "preference", "metric": "ties", "value": ties},
        {"section": "preference", "metric": "full_pacer_win_proportion_non_ties", "value": wins / (wins + losses)},
        {"section": "preference", "metric": "exact_two_sided_binomial_sign_test_p", "value": sign_p},
    ]
    score_rows = []
    for score in SCORES:
        delta = f"delta_{score}"
        primary[delta] = primary[f"full_pacer_{score}"] - primary[f"content_only_{score}"]
        low, high, improved, tied, worse = cluster_ci(primary, delta)
        score_rows.append({
            "score": score,
            "content_only_mean": primary[f"content_only_{score}"].mean(),
            "full_pacer_mean": primary[f"full_pacer_{score}"].mean(),
            "paired_delta": primary[delta].mean(),
            "cluster_bootstrap_CI_low": low, "cluster_bootstrap_CI_high": high,
            "lectures_improved": improved, "lectures_tied": tied, "lectures_worse": worse,
            "bootstrap_replicates": 100_000, "seed": SEED,
        })
    summary_rows = preference_rows.copy()
    for score_row in score_rows:
        for metric, value in score_row.items():
            if metric != "score":
                summary_rows.append({"section": score_row["score"], "metric": metric, "value": value})
    pd.DataFrame(summary_rows).to_csv(judge_dir / f"{args.prefix}_primary_summary.csv", index=False)

    family_rows = []
    for task_type, group in primary.groupby("task_type", sort=True):
        row = {
            "task_family": task_type, "N": len(group),
            "full_pacer_wins": int((group.mapped_preference == "full_pacer").sum()),
            "ties": int((group.mapped_preference == "tie").sum()),
            "content_only_wins": int((group.mapped_preference == "content_only").sum()),
        }
        for score in SCORES:
            row[f"content_only_{score}_mean"] = group[f"content_only_{score}"].mean()
            row[f"full_pacer_{score}_mean"] = group[f"full_pacer_{score}"].mean()
            row[f"{score}_delta"] = (group[f"full_pacer_{score}"] - group[f"content_only_{score}"]).mean()
        family_rows.append(row)
    family = pd.DataFrame(family_rows)
    family.to_csv(judge_dir / f"{args.prefix}_by_task_family.csv", index=False)

    merged = primary.merge(swapped, on=["task_id", "task_type", "target_lecture"], suffixes=("_primary", "_swapped"), validate="one_to_one")
    non_tie = (merged.mapped_preference_primary != "tie") & (merged.mapped_preference_swapped != "tie")
    robustness_rows = [
        {"metric": "exact_three_way_preference_agreement", "value": (merged.mapped_preference_primary == merged.mapped_preference_swapped).mean(), "denominator": 600},
        {"metric": "winner_agreement_excluding_any_orientation_with_tie", "value": (merged.loc[non_tie, "mapped_preference_primary"] == merged.loc[non_tie, "mapped_preference_swapped"]).mean(), "denominator": int(non_tie.sum())},
        {"metric": "content_only_to_full_pacer_flip_rate", "value": ((merged.mapped_preference_primary == "content_only") & (merged.mapped_preference_swapped == "full_pacer")).mean(), "denominator": 600},
        {"metric": "full_pacer_to_content_only_flip_rate", "value": ((merged.mapped_preference_primary == "full_pacer") & (merged.mapped_preference_swapped == "content_only")).mean(), "denominator": 600},
    ]
    for pass_name, frame in (("primary", primary), ("swapped", swapped)):
        non_ties = frame.overall_preference != "TIE"
        robustness_rows.append({"metric": f"A_position_win_rate_{pass_name}_among_non_ties", "value": (frame.loc[non_ties, "overall_preference"] == "A").mean(), "denominator": int(non_ties.sum())})
    for score in SCORES:
        for condition in ("content_only", "full_pacer"):
            x = merged[f"{condition}_{score}_primary"]
            y = merged[f"{condition}_{score}_swapped"]
            correlation = pearsonr(x, y).statistic if x.nunique() > 1 and y.nunique() > 1 else np.nan
            robustness_rows.append({"metric": f"pearson_{condition}_{score}_across_orientations", "value": correlation, "denominator": 600})
    robustness = pd.DataFrame(robustness_rows)
    robustness.to_csv(judge_dir / f"{args.prefix}_order_robustness.csv", index=False)

    keep = ["task_id", "task_type", "target_lecture"]
    for pass_name in ("primary", "swapped"):
        keep.extend([f"mapped_preference_{pass_name}"] + [f"{condition}_{score}_{pass_name}" for condition in ("content_only", "full_pacer") for score in SCORES])
    merged[keep].to_csv(judge_dir / f"{args.prefix}_task_level_summary_600.csv", index=False)

    report = f"# {args.display_name} automated-judge report\n\n"
    report += "Primary inference uses only the first randomized orientation. The complete exact-swapped pass is an order-sensitivity audit and is not combined with the primary pass.\n\n"
    report += "## Primary preference\n\n"
    report += f"Full PACER wins: {wins}; content-only wins: {losses}; ties: {ties}. Full-PACER non-tie win proportion: {wins/(wins+losses):.6f}. Exact two-sided binomial sign-test p={sign_p:.8g}. This is automated-judge preference, not human preference.\n\n"
    report += "## Absolute scores (primary pass only)\n\n" + markdown_table(pd.DataFrame(score_rows)) + "\n\n"
    report += "Intervals are percentile 95% target-lecture-cluster bootstrap intervals using 100,000 replicates and seed 20260919. Lecture improved/tied/worse counts compare lecture-level paired mean differences.\n\n"
    report += "## Task-family breakdown (descriptive)\n\n" + markdown_table(family) + "\n\nNo family-level significance tests were added.\n\n"
    report += "## Complete order-swap robustness\n\n" + markdown_table(robustness) + "\n\nNo voting rule combines the orientations.\n\n"
    family_note = ("The local Qwen3.6 judge shares the Qwen model family with the Qwen3-8B generator and is secondary same-family robustness evidence."
                   if args.prefix == "qwen36" else
                   "OpenAI GPT-5.6 Sol does not share the Qwen generator family and is the primary independent-family automated judge.")
    report += f"## Scientific limitations\n\n1. This is an automated judge, not a human evaluation.\n2. The judge evaluates agreement with the benchmark reference answer and supplied evidence, not independent clinical truth.\n3. {family_note}\n4. Only one checkpoint is represented in this per-judge report.\n5. The result does not demonstrate learning outcomes or clinical safety.\n6. The result supplements rather than replaces lexical-F1 and MPNet semantic-similarity analyses.\n"
    (judge_dir / args.report_filename).write_text(report, encoding="utf-8")
    print(f"{args.prefix.upper()}_ANALYSIS_COMPLETE = PASS wins={wins} content={losses} ties={ties}")


if __name__ == "__main__":
    main()
