# Experiment Map

This document maps the public code to the major analyses reported in the manuscript.

## Core Benchmark

```text
src/benchmark/
```

The benchmark code supports course-grounded QA, evidence retrieval, attribution, and sequence-policy evaluation.

## Evidence Delivery

```text
src/retrieval/
```

The retrieval analyses include:

- BM25
- reciprocal-rank fusion
- dense retrieval
- hierarchical lecture-to-slide retrieval
- cross-encoder reranking
- course-position-constrained retrieval

## Course-Aware Ranking

```text
src/ranking/run_course_position_ranker.py
```

The valid course-aware ranker uses only leakage-safe lecture-level position signals. Exact target-position distance is excluded.

The two reported course-aware ranking systems are:

1. RRF plus a coarse lecture prior.
2. A grouped logistic ranker using content-ranking features, `same_lecture`, and `lecture_distance`.

## Matched Generation

```text
src/generation/
```

Generation uses frozen retrieval contexts and deterministic decoding (`do_sample=False`) under the model-specific chat serialization implemented in the released code.

## Exact Source Attribution

```text
src/attribution/run_exact_citation_audit.py
src/common/citation_metrics.py
```

Evidence delivery and exact source attribution are evaluated separately.

## Sequence and Scope Control

```text
src/controller/
```

The controller analyses include:

- course-support control
- concept-held-out evaluation
- feature ablation
- answerable-task false-control analysis
- alternate-surface-form robustness

## Retrieval-Generation Linkage

```text
src/evaluation/run_retrieval_generation_linkage.py
```

This analysis links retrieval conditions to downstream answer-quality and attribution outcomes.

## Policy Analysis

```text
src/evaluation/run_policy_control_analysis.py
```

This component evaluates policy behavior after course-support control.

## External Validation

```text
src/external/run_external_course_aware_controller.py
```

External validation is reported separately from the within-course benchmark and is not pooled with it.
