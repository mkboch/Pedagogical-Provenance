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
