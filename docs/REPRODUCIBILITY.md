# Reproducibility

This repository is the curated public code release for PACER.

The release intentionally separates source code and small redistributable
benchmark inputs from generated results and source-derived course data.

## Released evaluation sets

    data/benchmark/canonical_multiconcept_sequence.jsonl
    data/benchmark/alternate_form_sequence.jsonl

These contain 228 canonical multi-concept tasks and 96 matched
alternate-surface-form tasks.

## Inputs not redistributed

The following are not included in Git:

- raw course materials
- the 1,000-task core QA benchmark, whose reference answers contain
  source-derived instructional text
- generated model responses
- experimental result tables
- candidate-feature matrices and other generated intermediates
- model weights and checkpoints
- caches
- private validation packages

The released code documents the expected local paths for these inputs.

## Generated outputs

Runtime outputs belong under:

    artifacts/

The directory is ignored by Git except for its placeholder file.

Generated results should not be committed to the repository.

## Main experiment map

See docs/EXPERIMENT_MAP.md.

## Post-freeze verification

Two targeted verification utilities are provided under:

    src/audit/

They cover the lecture-prior scale/boundary analysis and the matched
canonical/alternate paired-rescore summary.

See docs/POSTFREEZE_AUDITS.md.

These utilities consume local frozen intermediates. They do not redistribute
those intermediates or generated audit results.

## Integrity verification

Run:

    sha256sum -c CODE_CHECKSUMS.sha256
    sha256sum -c BENCHMARK_CHECKSUMS.sha256

Every released Python source file and both released sequence-policy benchmark
inputs should return OK.


## Final-analysis scripts

The release includes source code for the final matched logistic ablation,
nested-CV LambdaMART comparison, metadata-reliability analyses, and paired
Qwen3 generation/evaluation.

These analyses require locally generated upstream files under artifacts.
Those files are not committed because they include generated outputs,
candidate-feature matrices, or dependencies derived from course materials
that are not redistributed.

The public copies preserve the frozen scoring and statistical logic while
replacing machine-specific absolute paths with repository-local paths.

Source hashes for the frozen scripts are recorded in
docs/FINAL_ANALYSIS_PROVENANCE.md.

The final semantic evaluation used
sentence-transformers/all-mpnet-base-v2 with cached revision
e8c3b32edf5434bc2275fc9bab85f82640a19130.
