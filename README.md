# PACER

Research code for:

**PACER: Pedagogical Alignment and Course-Aware Evidence Ranking for Educational RAG in Medical Imaging**

## Overview

PACER studies whether coarse learner/course position can improve the ordering
of instructional evidence in educational retrieval-augmented generation.

In this work, pedagogical alignment has a deliberately narrow operational
meaning: evidence is aligned with the learner's known course position. It does
not mean that learner outcomes or instructional effectiveness were measured.

The repository supports the component-level evaluation reported in the
manuscript across:

- evidence delivery
- leakage-safe course-aware evidence ranking
- exact instructional-source attribution
- sequence and scope control
- targeted post-freeze verification analyses

## Release Scope

This repository releases source code, documentation, and two sequence-policy
evaluation sets.

Released benchmark inputs:

- data/benchmark/canonical_multiconcept_sequence.jsonl
  - 228 canonical multi-concept sequence-policy tasks
- data/benchmark/alternate_form_sequence.jsonl
  - 96 matched alternate-surface-form sequence-policy tasks

Their SHA-256 checksums are recorded in BENCHMARK_CHECKSUMS.sha256.

The repository does not redistribute:

- course slides or lecture text
- the 1,000-task core QA benchmark, whose reference answers contain
  source-derived instructional text
- generated model answers
- experimental result tables
- candidate-feature matrices and other generated intermediates
- model weights or model caches
- private validation packages
- manuscript source or PDFs

Generated runtime outputs belong under artifacts, which is ignored by Git
except for its placeholder file.

## Repository Structure

    Pedagogical-Provenance/
    ├── README.md
    ├── requirements.txt
    ├── CODE_CHECKSUMS.sha256
    ├── BENCHMARK_CHECKSUMS.sha256
    ├── configs/
    ├── docs/
    │   ├── DATA_LAYOUT.md
    │   ├── EXPERIMENT_MAP.md
    │   ├── POSTFREEZE_AUDITS.md
    │   └── REPRODUCIBILITY.md
    ├── data/
    │   ├── benchmark/
    │   ├── corpus/
    │   └── external/
    ├── artifacts/
    └── src/
        ├── attribution/
        ├── audit/
        ├── benchmark/
        ├── common/
        ├── controller/
        ├── evaluation/
        ├── external/
        ├── generation/
        ├── ranking/
        └── retrieval/

## Main Experimental Components

### Benchmark construction

    src/benchmark/build_core_benchmark_audit.py
    src/benchmark/build_canonical_multiconcept_sequence_set.py
    src/benchmark/build_alternate_form_sequence_set.py

### Evidence delivery and retrieval

    src/retrieval/run_position_constrained_retrieval.py
    src/retrieval/run_matched_retrieval_core.py
    src/retrieval/run_advanced_retrieval.py
    src/retrieval/run_advanced_retrieval_diagnostics.py
    src/retrieval/run_rrf_lecture_slide_hybrid.py
    src/retrieval/run_cross_encoder_diagnostics.py
    src/retrieval/run_stronger_cross_encoder.py
    src/retrieval/run_full_cross_encoder_evaluation.py

### Leakage-safe course-aware ranking

    src/ranking/run_course_position_ranker.py

The benchmark current course position is anchored to the target instructional
location. Exact target-slide position distances would therefore identify the
target directly and are excluded from the reported leakage-safe ranker.

Excluded features:

    signed_position_distance
    abs_position_distance
    slide_distance_same_lecture

Allowed coarse position features:

    same_lecture
    lecture_distance

Cross-validation is grouped by target lecture.

### Matched generation

    src/generation/run_matched_global_position_generation.py
    src/generation/merge_matched_generation_results.py

### Exact source attribution

    src/attribution/run_exact_citation_audit.py
    src/common/citation_metrics.py

### Course-support control and robustness

    src/controller/run_course_aware_controller.py
    src/controller/run_concept_held_out_evaluation.py
    src/controller/run_controller_ablation.py
    src/controller/run_position_controller_analysis.py
    src/controller/run_controller_surface_form_robustness.py

### Retrieval-generation and policy analysis

    src/evaluation/run_retrieval_generation_linkage.py
    src/evaluation/run_policy_control_analysis.py

### External validation

    src/external/run_external_course_aware_controller.py

### Post-freeze verification utilities

    src/audit/run_lecture_prior_boundary_audit.py
    src/audit/summarize_paired_rescore.py

See docs/POSTFREEZE_AUDITS.md.

These utilities operate on locally generated frozen artifacts. The generated
audit outputs themselves are not committed.

## Installation

    python -m venv .venv
    source .venv/bin/activate
    python -m pip install --upgrade pip
    pip install -r requirements.txt

GPU-backed generation and embedding experiments require a CUDA-compatible
PyTorch installation appropriate for the local system.

## Local Data

The non-redistributed core QA benchmark is expected locally at:

    data/benchmark/core_qa_benchmark.jsonl

The local slide corpus is expected under:

    data/corpus/

External validation inputs, when available, belong under:

    data/external/

See docs/DATA_LAYOUT.md.

## Reproducibility

See:

- docs/EXPERIMENT_MAP.md
- docs/DATA_LAYOUT.md
- docs/REPRODUCIBILITY.md
- docs/POSTFREEZE_AUDITS.md

Some analyses require upstream artifacts generated from course materials that
cannot be redistributed. Those dependencies are documented rather than
silently replaced with substitute data.

## Source Integrity

SHA-256 checksums for released Python files are stored in:

    CODE_CHECKSUMS.sha256

Checksums for the two released sequence-policy evaluation sets are stored in:

    BENCHMARK_CHECKSUMS.sha256

Verify with:

    sha256sum -c CODE_CHECKSUMS.sha256
    sha256sum -c BENCHMARK_CHECKSUMS.sha256

## Citation

A formal citation entry will be added after the manuscript receives a
permanent publication record.

## Contact

For questions about the code or reproducibility, please open an issue in this
repository.
