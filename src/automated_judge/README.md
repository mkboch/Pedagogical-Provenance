# Automated-judge reproducibility code

This directory contains the public, path-portable code for the blinded automated-judge analysis reported in the PACER manuscript. No benchmark rows, course text, generated answers, judge inputs, raw judge responses, model weights, API credentials, or result tables are included.

The reported analysis used:

- generator: `Qwen/Qwen3-8B`;
- local robustness judge: `Qwen/Qwen3.6-35B-A3B`, revision `995ad96eacd98c81ed38be0c5b274b04031597b0`;
- cross-family API judge: `gpt-5.6-sol`;
- randomization/bootstrap seed: `20260919`;
- 600 primary and 600 exact-swapped judgments per judge;
- the rubric in `common_judge_prompt.txt`.

The scripts preserve the frozen scoring, blinding, order-swap, and statistical logic while using repository-local working paths. Generated audit files are written under `artifacts/automated_judge/`, which is not part of the source release. The frozen generation inputs and outputs required to reconstruct the judge-visible pairs are not redistributed because they contain source-derived instructional text and model outputs.

`OPENAI_API_KEY` is read only from the environment by the API scripts. No credential is stored in this repository.

## Workflow

1. Place authorized local copies of the frozen generation inputs and outputs under `outputs/jiis_final_experiments_20260918/`. The scripts verify their published SHA-256 hashes before use.
2. Run `code/build_clean_inputs.py` to construct the blinded primary and exact-swapped inputs. It verifies that no substantive question, reference, evidence, or answer text is modified.
3. Run `code/run_qwen36_judge.py` with the pinned local Qwen3.6 snapshot.
4. Use the OpenAI Batch helpers to construct, cost-check, submit, collect, and parse the GPT-5.6 Sol judgments.
5. Run `code/analyze_single_judge.py` for each judge and `code/analyze_cross_judge.py` for the primary-orientation cross-judge analysis.

The primary inferential orientation is the first randomized A/B assignment. The complete swapped orientation is used only to quantify order sensitivity and is not combined with the primary pass by voting.
