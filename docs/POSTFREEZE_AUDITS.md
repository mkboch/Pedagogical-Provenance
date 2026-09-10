# Post-freeze verification utilities

PACER includes targeted post-freeze checks for two claims that required
additional verification after the main experimental protocol was frozen.

These utilities are diagnostic verification code. They do not change the
frozen experimental protocol and they do not redistribute generated result
tables.

## Lecture-prior scale and boundary audit

The released course-aware ranking score is:

    rrf_rr + w * (same_lecture - 0.01 * lecture_distance)

Here rrf_rr is the reciprocal of the final fused RRF rank.

Example command:

    python src/audit/run_lecture_prior_boundary_audit.py \
      --candidate-features /path/to/position_aware_candidate_features.csv

The audit:

- evaluates the complete frozen grid:
  0, 0.05, 0.10, 0.20, 0.40, 0.80, 1.60
- evaluates post-freeze diagnostic points 3.20 and 6.40
- derives the sufficient strict same-lecture block threshold
- counts tasks with at least three same-lecture candidates
- reproduces five-fold GroupKFold weight selection
- compares the frozen additive score with a parameter-free rule:
  same lecture first, then original RRF rank

The frozen candidate-feature artifact is not committed because it is a
generated experimental intermediate.

The frozen local candidate-feature artifact used for the manuscript audit had
SHA-256:

    816c04bbf7d32d08592f20e42f01caca82c963aa44dedddc8946750c92854886

## Paired canonical/alternate rescore verification

Example command:

    python src/audit/summarize_paired_rescore.py \
      --paired-predictions /path/to/paired_96_exact_predictions.csv

The utility independently recomputes:

- canonical and alternate matched accuracy
- available and sequence strata
- exact-lookup accuracy
- discordant-pair counts
- nesting of alternate-correct cases within canonical-correct cases
- the exact two-sided McNemar test

The paired prediction CSV is a generated result artifact and is not
redistributed.

The full deterministic reconstruction also included an upstream check that
the reconstructed canonical out-of-fold predictions agreed with the retained
228-item predictions.

That upstream gate depends on non-redistributed frozen intermediate features
and predictions. The public paired utility therefore verifies the matched
96-pair rescore table but does not claim to regenerate the separate 228-item
feature reconstruction from raw course data.

## Release policy

The repository tracks:

- source code
- documentation
- configuration information
- the two small released sequence-policy benchmark inputs

The repository does not track:

- generated result CSV or JSON files
- candidate-feature matrices
- generated model responses
- manuscript PDFs or LaTeX source
- raw course materials
- model weights or caches
- private validation packages
