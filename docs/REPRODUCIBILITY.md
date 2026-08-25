# Reproducibility

## Release Model

This repository releases source code and documentation rather than experimental outputs.

The following are intentionally excluded from Git:

- raw course materials
- benchmark instances
- generated model responses
- result CSV files
- model weights
- checkpoints
- caches
- private validation packages

## Environment

Install the Python dependencies with:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

GPU experiments require a CUDA-compatible PyTorch installation suitable for the local system.

## Local Paths

Repository-relative paths are used throughout the public release.

Primary local inputs:

```text
data/corpus/
data/benchmark/
data/external/
```

Generated outputs:

```text
artifacts/
```

## Pipeline Organization

The major pipeline stages are:

```text
benchmark construction
        ↓
retrieval and matched context construction
        ↓
course-aware ranking
        ↓
matched generation
        ↓
source attribution
        ↓
retrieval-generation linkage
        ↓
course-support control and robustness
        ↓
external validation
```

Some downstream scripts require locally generated outputs from upstream stages.

## Leakage-Safe Ranking

The public ranking implementation explicitly excludes exact slide-position-distance features.

The reported course-aware ranking result is implemented in:

```text
src/ranking/run_course_position_ranker.py
```

The logistic feature set consists of content-ranking features plus:

```text
same_lecture
lecture_distance
```

Five-fold cross-validation is grouped by target lecture.

## Prompt Construction

The exact context formatter and prompt templates used by the public retrieval pipeline are provided in:

```text
src/common/prompting.py
src/common/runtime_reference.py
```

The maximum context length used by the matched retrieval protocol is 650 words.

## Citation Parsing

The exact validated citation parser is released as code at:

```text
src/common/citation_metrics.py
```

It is used by:

```text
src/attribution/run_exact_citation_audit.py
```

## Integrity Verification

Run:

```bash
sha256sum -c CODE_CHECKSUMS.sha256
```

Every released Python source file should return `OK`.

## Numerical Results

The manuscript is the authoritative source for final numerical results and statistical interpretation.

No result tables are committed to this repository.
