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


## Post-freeze Verification

    src/audit/run_lecture_prior_boundary_audit.py
    src/audit/summarize_paired_rescore.py

The lecture-prior utility evaluates the frozen and extended diagnostic weight
grids, the strict lecture-block threshold, grouped fold selection, and the
parameter-free same-lecture-first comparator.

The paired-rescore utility independently recomputes the matched canonical and
alternate summary statistics and exact McNemar test from a locally generated
paired prediction table.

Generated audit tables are not tracked by Git. See
docs/POSTFREEZE_AUDITS.md.


## Final Manuscript Table-to-Code Map

### Table 2: matched ranking ablation and nonlinear control

    src/ranking/run_position_feature_ablation.py
    src/evaluation/run_cluster_aware_inference.py
    src/ranking/run_lambdamart_ablation.py

The first two scripts implement the matched logistic feature-family analysis
and lecture-clustered inference. The LambdaMART script implements nested
target-lecture-grouped selection and held-out evaluation for the content-only
and full PACER feature sets.

### Table 3: metadata-reliability operating envelope

    src/ranking/run_one_lecture_metadata_stress.py
    src/ranking/run_metadata_reliability.py
    src/ranking/run_lambdamart_metadata_stress.py

The logistic and LambdaMART stress analyses perturb test-time course-position
features by one lecture while retaining the fixed candidate pool.

### Table 5: paired Qwen3 end-to-end comparison

    src/generation/prepare_lambdamart_generation_inputs.py
    src/generation/run_qwen3_lambdamart_generation.py
    src/evaluation/evaluate_lambdamart_qwen3_generation.py
    src/common/final_generation_reference.py

The preparation script constructs the matched content-only and full-PACER
evidence contexts. The generation script performs deterministic Qwen3-8B
generation, and the evaluator performs the paired lexical and semantic
analysis with target-lecture-clustered inference.

Generated result files are intentionally excluded from the repository.
