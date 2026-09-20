#!/usr/bin/env python3
"""Primary-orientation-only Qwen3.6 versus OpenAI GPT-5.6 Sol comparison."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr


SCORES = ["correctness", "completeness_relevance", "evidence_faithfulness"]
CONDITIONS = ["content_only", "full_pacer"]
CATEGORIES = ["full_pacer", "content_only", "tie"]


def markdown_table(frame: pd.DataFrame) -> str:
    display = frame.copy()
    for column in display.select_dtypes(include=["float"]).columns:
        display[column] = display[column].map(lambda value: "" if pd.isna(value) else f"{value:.6f}")
    columns = list(display.columns)
    lines = ["| " + " | ".join(columns) + " |", "|" + "|".join(["---"] * len(columns)) + "|"]
    for row in display.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(str(value).replace("|", "\\|") for value in row) + " |")
    return "\n".join(lines)


def main() -> None:
    repo_root = Path(__file__).resolve().parents[3]
    root = repo_root / "artifacts" / "automated_judge"
    out = root / "cross_judge"
    out.mkdir(parents=True, exist_ok=True)
    qwen = pd.read_csv(root / "qwen36/qwen36_primary_parsed_600.csv")
    openai = pd.read_csv(root / "openai_gpt56_sol/openai_primary_parsed_600.csv")
    merged = qwen.merge(openai, on=["task_id", "task_type", "target_lecture"], suffixes=("_qwen36", "_openai"), validate="one_to_one")
    if len(merged) != 600:
        raise RuntimeError("Cross-judge merge did not yield 600 tasks")

    exact_agreement = float((merged.mapped_preference_qwen36 == merged.mapped_preference_openai).mean())
    observed = exact_agreement
    expected = sum(
        float((merged.mapped_preference_qwen36 == category).mean())
        * float((merged.mapped_preference_openai == category).mean())
        for category in CATEGORIES
    )
    kappa = (observed - expected) / (1 - expected) if expected < 1 else np.nan
    neither_tie = (merged.mapped_preference_qwen36 != "tie") & (merged.mapped_preference_openai != "tie")
    winner_agreement = float((merged.loc[neither_tie, "mapped_preference_qwen36"] == merged.loc[neither_tie, "mapped_preference_openai"]).mean())
    counts = {
        "both_prefer_full_pacer": int(((merged.mapped_preference_qwen36 == "full_pacer") & (merged.mapped_preference_openai == "full_pacer")).sum()),
        "both_prefer_content_only": int(((merged.mapped_preference_qwen36 == "content_only") & (merged.mapped_preference_openai == "content_only")).sum()),
        "qwen_full_openai_content": int(((merged.mapped_preference_qwen36 == "full_pacer") & (merged.mapped_preference_openai == "content_only")).sum()),
        "qwen_content_openai_full": int(((merged.mapped_preference_qwen36 == "content_only") & (merged.mapped_preference_openai == "full_pacer")).sum()),
        "one_or_both_tie": int(((merged.mapped_preference_qwen36 == "tie") | (merged.mapped_preference_openai == "tie")).sum()),
    }
    summary_rows = [
        {"metric": "three_way_preference_agreement", "value": exact_agreement, "denominator": 600},
        {"metric": "cohen_kappa_three_way", "value": kappa, "denominator": 600},
        {"metric": "winner_agreement_neither_ties", "value": winner_agreement, "denominator": int(neither_tie.sum())},
        *[{"metric": key, "value": value, "denominator": 600} for key, value in counts.items()],
    ]
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(out / "cross_judge_preference_summary.csv", index=False)

    correlation_rows = []
    for score in SCORES:
        for condition in CONDITIONS:
            qcol = f"{condition}_{score}_qwen36"
            ocol = f"{condition}_{score}_openai"
            correlation = pearsonr(merged[qcol], merged[ocol]).statistic if merged[qcol].nunique() > 1 and merged[ocol].nunique() > 1 else np.nan
            correlation_rows.append({
                "score": score, "condition": condition,
                "pearson_qwen36_vs_openai": correlation, "N": 600,
            })
    correlations = pd.DataFrame(correlation_rows)
    correlations.to_csv(out / "cross_judge_score_correlations.csv", index=False)

    effect_rows = []
    for score in SCORES:
        qdelta = float((qwen[f"full_pacer_{score}"] - qwen[f"content_only_{score}"]).mean())
        odelta = float((openai[f"full_pacer_{score}"] - openai[f"content_only_{score}"]).mean())
        effect_rows.append({
            "score": score, "qwen36_paired_delta": qdelta,
            "openai_gpt56_sol_paired_delta": odelta,
            "qwen36_positive_delta": qdelta > 0,
            "openai_positive_delta": odelta > 0,
        })
    effects = pd.DataFrame(effect_rows)
    effects.to_csv(out / "cross_judge_effects.csv", index=False)
    direction = pd.DataFrame([
        {"judge": "Qwen3.6", **{f"positive_{row.score}_delta": bool(row.qwen36_positive_delta) for row in effects.itertuples()},
         "more_full_pacer_wins_than_content_only": int((qwen.mapped_preference == "full_pacer").sum()) > int((qwen.mapped_preference == "content_only").sum())},
        {"judge": "OpenAI GPT-5.6 Sol", **{f"positive_{row.score}_delta": bool(row.openai_positive_delta) for row in effects.itertuples()},
         "more_full_pacer_wins_than_content_only": int((openai.mapped_preference == "full_pacer").sum()) > int((openai.mapped_preference == "content_only").sum())},
    ])
    direction.to_csv(out / "cross_judge_directional_checks.csv", index=False)
    merged[["task_id", "task_type", "target_lecture", "mapped_preference_qwen36", "mapped_preference_openai"]].to_csv(out / "cross_judge_task_preferences_600.csv", index=False)

    report = "# Qwen3.6 versus OpenAI GPT-5.6 Sol cross-judge robustness\n\n"
    report += "This comparison uses the primary randomized orientation only. OpenAI GPT-5.6 Sol remains the primary independent-family automated judge; Qwen3.6 is secondary same-family local robustness evidence. Neither is human evaluation.\n\n"
    report += "## Preference agreement\n\n" + markdown_table(summary) + "\n\n"
    report += "## Absolute-score correlations\n\n" + markdown_table(correlations) + "\n\n"
    report += "## Paired effects side by side\n\n" + markdown_table(effects) + "\n\n"
    report += "## Directional checks\n\n" + markdown_table(direction) + "\n\n"
    report += "The judges are not averaged, their p-values are not pooled, and no majority-vote endpoint is constructed. Disagreement is retained rather than selecting the more favorable judge.\n"
    (out / "CROSS_JUDGE_REPORT.md").write_text(report, encoding="utf-8")
    print(f"CROSS_JUDGE_ANALYSIS = PASS agreement={exact_agreement:.6f} kappa={kappa:.6f}")


if __name__ == "__main__":
    main()
