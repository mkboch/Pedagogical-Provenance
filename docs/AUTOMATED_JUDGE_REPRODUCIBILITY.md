# Automated-judge reproducibility

The PACER downstream evaluation uses two blinded automated judges over the same 600 paired Qwen3-8B outputs. OpenAI GPT-5.6 Sol is the primary cross-family automated judge; `Qwen/Qwen3.6-35B-A3B` at revision `995ad96eacd98c81ed38be0c5b274b04031597b0` is a secondary same-family robustness judge. Neither evaluation is a human study.

The common pipeline is released under `src/automated_judge/`. A fixed seed (`20260919`) assigns the content-only and full-PACER conditions to anonymous A/B labels for the primary pass. The swapped pass reverses A/B for every task. Questions, reference answers, evidence contexts, and generated answers are copied without substantive text modification; system identities are withheld from the judge-visible payload.

Each judge independently scores correctness, completeness/relevance, and evidence faithfulness on 1--5 scales and returns an overall A/B/TIE preference. Primary statistics use only the first randomized orientation. The swapped orientation quantifies order sensitivity. Target-lecture-cluster bootstrap intervals use 100,000 replicates with seed `20260919`. Cross-judge agreement is reported descriptively; judge scores and p-values are not pooled.

The OpenAI scripts use the Responses-compatible Batch API, structured JSON output, `reasoning.effort=none`, and a 220-token output cap. API credentials are read from `OPENAI_API_KEY` and are never written by the released code. The local Qwen3.6 runner uses non-thinking deterministic structured decoding.

The source-derived judge inputs, raw model responses, generated Qwen answers, candidate matrices, and model weights are not redistributed. Replaying the complete analysis therefore requires authorized local copies of the frozen non-redistributed inputs whose hashes are checked by the scripts.
