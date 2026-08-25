# Local Data and Artifact Layout

The public repository does not contain experimental data or results.

## Instructional Corpus

Place the course corpus at:

```text
data/corpus/slide_corpus_final.jsonl
```

Expected document fields include:

```text
doc_id
lecture_id
slide_id
text
```

Some scripts additionally use an explicit course-position field when available.

## Core Benchmark

Place the core benchmark at:

```text
data/benchmark/core_qa_benchmark.jsonl
```

Expected fields include:

```text
task_id
task_type
question
target_doc_id
evidence_doc_ids
```

Sequence-policy builders may create additional local JSONL files under `artifacts/sequence_policy/`.

## External Validation

Place external validation resources under:

```text
data/external/
```

For the lecture-presentation external evaluation, the code expects the corresponding task and ordered-corpus files under a local `lpm/` subdirectory.

## Generated Artifacts

All generated files belong under:

```text
artifacts/
```

Typical local subdirectories include:

```text
artifacts/
├── matched_retrieval/
├── matched_generation/
├── retrieval_diagnostics/
├── course_position_ranking/
├── citation_evaluation/
├── sequence_policy/
├── controller_analysis/
└── external_validation/
```

The entire artifact tree is ignored by Git except for `artifacts/.gitignore`.

## Upstream Reference Artifacts

Some analysis scripts validate or compare against upstream files produced by earlier pipeline stages. These files are local computational dependencies and are not part of the public Git repository.

They should be placed under the documented `artifacts/` paths before running the dependent analysis.
