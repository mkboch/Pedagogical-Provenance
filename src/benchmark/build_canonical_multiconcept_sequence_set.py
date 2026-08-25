#!/usr/bin/env python3
"""Build Canonical Multi-Concept Sequence Set.

SEPARATE artifact. Does NOT touch Core QA Benchmark (which remains immutable).

Motivation (see 01_BENCHMARK_V1_SEQUENCE_CONCEPT_AUDIT.md): Core QA Benchmark's
200 lecture_sequence_violation_probe tasks use exactly TWO future concepts
("ultrasound", "k-space"), assigned by the hard-coded rule
`future = "ultrasound" if lecture_id < 10 else "k-space"`
(src/rebuild_minimal_pipeline.py:274). Concept identity is therefore a
deterministic function of lecture position, and every v1 sequence probe is
single-class (all "not yet available"). v1 remains a valid narrow
stress test, but cannot demonstrate arbitrary-concept generalization.

This set addresses both limitations deterministically:
  - many concepts (not 2), each with a corpus-derived first-occurrence
  - BOTH classes ("available" / "not_yet_available") under the SAME
    question template, so the label is a function of the
    concept x anchor-position interaction, never of the template or of
    the lecture range alone.

No LLM is used. No randomness except an explicitly seeded balanced
subsample. Every label is deterministically reconstructable from
(concept first-occurrence position, anchor position).
"""

from pathlib import Path
import hashlib
import json
import re
import unicodedata

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "artifacts" / "sequence_policy"
OUT.mkdir(parents=True, exist_ok=True)

CORPUS = ROOT / "data/corpus/slide_corpus_final.jsonl"
SEED = 20260822

# Candidate concept vocabulary. Sourced verbatim from the DOMAIN list in
# src/rebuild_minimal_pipeline.py:150-158 (the project's own, already-used
# imaging concept vocabulary) so that no medical concept is invented here.
DOMAIN_VOCAB = [
    "medical imaging", "x-ray", "radiography", "computed tomography", "ct", "tomography",
    "mri", "ultrasound", "pet", "spect", "fourier transform", "fourier series",
    "fourier analysis", "sampling", "aliasing", "convolution", "filtering",
    "projection", "sinogram", "radon transform", "backprojection",
    "filtered backprojection", "reconstruction", "attenuation", "detector",
    "contrast", "resolution", "noise", "dose", "matlab", "image quality",
    "inner product", "orthogonal", "delta function", "impulse response",
    "linear transformation", "linear system",
]

# Additional concepts appearing in the v1 sequence family, retained so the
# set broadens concept coverage.
DOMAIN_VOCAB += ["k-space"]

# Concepts too generic to have a defensible "first introduction" point
# (they appear in essentially every lecture as ordinary vocabulary rather
# than as a taught concept). Excluded by a prespecified rule below
# (occurrence_lecture_fraction threshold), not hand-picked per concept.


def norm(s):
    s = unicodedata.normalize("NFKC", str(s or "")).lower()
    s = s.replace("‐", "-").replace("‑", "-").replace("–", "-").replace("—", "-")
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def phrase_pattern(concept):
    """Word-boundary, hyphen/space-flexible exact phrase matcher."""
    parts = [re.escape(p) for p in norm(concept).replace("-", " ").split()]
    return re.compile(r"(?<![a-z0-9])" + r"[\s\-]+".join(parts) + r"(?![a-z0-9])")


def main():
    corpus = [json.loads(l) for l in CORPUS.read_text(encoding="utf-8").splitlines() if l.strip()]
    if len(corpus) != 1117:
        raise SystemExit(f"FAILED: expected 1117 corpus docs, got {len(corpus)}")

    for d in corpus:
        d["lecture_id"] = int(d["lecture_id"])
        d["slide_id"] = int(d["slide_id"])
        d["_norm"] = norm(d.get("text", ""))

    # Global course position: strict (lecture, slide) ordering.
    ordered = sorted(corpus, key=lambda d: (d["lecture_id"], d["slide_id"]))
    for i, d in enumerate(ordered):
        d["course_position"] = i
    by_doc = {d["doc_id"]: d for d in ordered}
    n_lectures = len({d["lecture_id"] for d in ordered})

    # ---------------------------------------------------------------
    # CONCEPT INVENTORY: first occurrence + spread, from corpus only
    # ---------------------------------------------------------------
    inv_rows = []
    concept_hits = {}
    for concept in sorted(set(DOMAIN_VOCAB)):
        pat = phrase_pattern(concept)
        hits = [d for d in ordered if pat.search(d["_norm"])]
        if not hits:
            continue
        first = hits[0]
        lectures_hit = sorted({d["lecture_id"] for d in hits})
        concept_hits[concept] = hits
        inv_rows.append(
            {
                "concept": concept,
                "n_occurrences": len(hits),
                "n_lectures_with_occurrence": len(lectures_hit),
                "lecture_coverage_fraction": len(lectures_hit) / n_lectures,
                "first_lecture": first["lecture_id"],
                "first_slide": first["slide_id"],
                "first_doc_id": first["doc_id"],
                "first_course_position": first["course_position"],
                "first_occurrence_evidence_snippet": re.sub(r"\s+", " ", first.get("text", ""))[:300],
                "lectures_with_occurrence": json.dumps(lectures_hit),
            }
        )

    inv = pd.DataFrame(inv_rows).sort_values("first_course_position").reset_index(drop=True)

    # ---------------------------------------------------------------
    # PRESPECIFIED ELIGIBILITY FILTERS (applied before any task built,
    # never tuned against downstream controller performance)
    # ---------------------------------------------------------------
    MIN_OCCURRENCES = 3          # recurs enough to be a taught concept
    MAX_LECTURE_COVERAGE = 0.80  # not omnipresent generic vocabulary
    MIN_PRE_SLIDES = 30          # enough course before it to anchor "not yet available"
    MIN_POST_SLIDES = 30         # enough course after it to anchor "available"

    inv["n_slides_before_first"] = inv["first_course_position"]
    inv["n_slides_after_first"] = len(ordered) - 1 - inv["first_course_position"]
    inv["eligible"] = (
        (inv["n_occurrences"] >= MIN_OCCURRENCES)
        & (inv["lecture_coverage_fraction"] <= MAX_LECTURE_COVERAGE)
        & (inv["n_slides_before_first"] >= MIN_PRE_SLIDES)
        & (inv["n_slides_after_first"] >= MIN_POST_SLIDES)
    )
    inv["exclusion_reason"] = ""
    inv.loc[inv["n_occurrences"] < MIN_OCCURRENCES, "exclusion_reason"] += "too_few_occurrences;"
    inv.loc[inv["lecture_coverage_fraction"] > MAX_LECTURE_COVERAGE, "exclusion_reason"] += "omnipresent_generic_vocab;"
    inv.loc[inv["n_slides_before_first"] < MIN_PRE_SLIDES, "exclusion_reason"] += "insufficient_pre_context;"
    inv.loc[inv["n_slides_after_first"] < MIN_POST_SLIDES, "exclusion_reason"] += "insufficient_post_context;"

    inv.to_csv(OUT / "sequence_generalization_concept_inventory.csv", index=False)
    eligible_concepts = inv[inv["eligible"]].copy()

    # ---------------------------------------------------------------
    # TASK CONSTRUCTION
    #
    # Same template for BOTH classes. Label depends ONLY on whether the
    # anchor position precedes the concept's first-occurrence position.
    # Anchors are sampled deterministically (fixed stride) from slides
    # that do NOT themselves mention the concept, so the anchor slide's
    # own text never leaks the answer.
    # ---------------------------------------------------------------
    TEMPLATE = ("At Lecture {lec}, Slide {sl}, explain {concept} using only "
                "concepts introduced up to that point in the course.")

    REF_UNAVAILABLE = (
        "A pedagogically faithful answer should use only material available up to "
        "Lecture {lec}, Slide {sl}. Because {concept} has not yet been introduced by "
        "that point in the course, the answer should state that the available lecture "
        "evidence does not support explaining it yet, rather than importing later "
        "lecture content."
    )
    REF_AVAILABLE = (
        "A pedagogically faithful answer should use only material available up to "
        "Lecture {lec}, Slide {sl}. Because {concept} has already been introduced by "
        "that point in the course, the answer should explain it using the course "
        "material available up to that position."
    )

    ANCHORS_PER_CLASS_PER_CONCEPT = 6
    tasks = []
    counter = 0

    for _, c in eligible_concepts.iterrows():
        concept = c["concept"]
        first_pos = int(c["first_course_position"])
        pat = phrase_pattern(concept)

        pre = [d for d in ordered if d["course_position"] < first_pos and not pat.search(d["_norm"])]
        post = [d for d in ordered if d["course_position"] > first_pos and not pat.search(d["_norm"])]

        for pool, label in ((pre, "not_yet_available"), (post, "available")):
            if len(pool) < ANCHORS_PER_CLASS_PER_CONCEPT:
                continue
            # deterministic evenly-spaced stride across the eligible pool
            stride = len(pool) / ANCHORS_PER_CLASS_PER_CONCEPT
            picks = [pool[int(i * stride)] for i in range(ANCHORS_PER_CLASS_PER_CONCEPT)]
            for a in picks:
                counter += 1
                ref = (REF_UNAVAILABLE if label == "not_yet_available" else REF_AVAILABLE).format(
                    lec=a["lecture_id"], sl=a["slide_id"], concept=concept
                )
                tasks.append(
                    {
                        "task_id": f"SGX_{counter:05d}",
                        "task_type": "sequence_generalization_probe",
                        "question": TEMPLATE.format(lec=a["lecture_id"], sl=a["slide_id"], concept=concept),
                        "reference_answer": ref,
                        "evidence_doc_ids": [a["doc_id"]],
                        "target_doc_id": a["doc_id"],
                        "metadata": {
                            "artifact_role": "1.1",
                            "concept": concept,
                            "concept_first_lecture": int(c["first_lecture"]),
                            "concept_first_slide": int(c["first_slide"]),
                            "concept_first_doc_id": c["first_doc_id"],
                            "concept_first_course_position": first_pos,
                            "anchor_lecture": a["lecture_id"],
                            "anchor_slide": a["slide_id"],
                            "anchor_course_position": a["course_position"],
                            "sequence_label": label,
                            "policy_class": (
                                "sequence_sensitive" if label == "not_yet_available" else "answerable"
                            ),
                        },
                    }
                )

    ext = pd.DataFrame(tasks)

    # ---------------------------------------------------------------
    # WRITE ARTIFACT
    # ---------------------------------------------------------------
    jsonl_path = OUT / "canonical_multiconcept_sequence.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as f:
        for t in tasks:
            f.write(json.dumps(t, ensure_ascii=False) + "\n")

    inv_hash = hashlib.sha256((OUT / "sequence_generalization_concept_inventory.csv").read_bytes()).hexdigest()
    ext_hash = hashlib.sha256(jsonl_path.read_bytes()).hexdigest()
    corpus_hash = hashlib.sha256(CORPUS.read_bytes()).hexdigest()

    labels = pd.Series([t["metadata"]["sequence_label"] for t in tasks]).value_counts().to_dict()
    concepts_used = sorted({t["metadata"]["concept"] for t in tasks})

    meta = {
        "version": "1.1",
        "artifact_name": "Canonical Multi-Concept Sequence Set",
        "creation_date": "2026-08-22",
        "relationship_to_core_benchmark": (
            "SEPARATE ARTIFACT. Core QA Benchmark (1000 tasks, sha256 "
            "8fa4c1a39149cbcf560778c50bc1c7189d22e94cc4e8baf1e1d1f041d3f9d4b2) is "
            "UNMODIFIED. These tasks are NOT merged into it."
        ),
        "source_corpus": str(CORPUS),
        "source_corpus_sha256": corpus_hash,
        "concept_inventory_sha256": inv_hash,
        "artifact_sha256": ext_hash,
        "task_count": len(tasks),
        "unique_concept_count": len(concepts_used),
        "concepts_used": concepts_used,
        "class_counts": labels,
        "anchors_per_class_per_concept": ANCHORS_PER_CLASS_PER_CONCEPT,
        "concept_vocabulary_source": "src/rebuild_minimal_pipeline.py:150-158 DOMAIN list (+ 'k-space'); no invented concepts",
        "eligibility_filters": {
            "min_occurrences": MIN_OCCURRENCES,
            "max_lecture_coverage_fraction": MAX_LECTURE_COVERAGE,
            "min_slides_before_first_occurrence": MIN_PRE_SLIDES,
            "min_slides_after_first_occurrence": MIN_POST_SLIDES,
        },
        "deterministic_construction_procedure": (
            "1) Normalize corpus text; order all 1117 slides by (lecture_id, slide_id) "
            "to give each a global course_position. 2) For each candidate concept from the "
            "project's own DOMAIN vocabulary, find all exact hyphen/space-flexible "
            "word-boundary phrase matches; the earliest is its first-occurrence position. "
            "3) Apply prespecified eligibility filters (above). 4) For each eligible concept, "
            "form a pre-pool (slides strictly before first occurrence) and post-pool (strictly "
            "after), each EXCLUDING slides that themselves mention the concept so the anchor "
            "slide never leaks the answer. 5) Select 6 anchors per pool by fixed even stride "
            "(no RNG). 6) Emit one task per anchor using an IDENTICAL question template for "
            "both classes; the label is determined solely by anchor_course_position < "
            "concept_first_course_position."
        ),
        "label_rule": "not_yet_available (policy_class=sequence_sensitive) iff anchor_course_position < concept_first_course_position; else available (policy_class=answerable)",
        "randomness": "NONE - fully deterministic even stride; SEED constant recorded for provenance only",
        "seed": SEED,
        "scripts": ["src/benchmark/build_canonical_multiconcept_sequence_set.py"],
        "no_llm_used": True,
        "no_human_ratings": True,
    }
    (OUT / "canonical_sequence_metadata.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )
    (OUT / "canonical_sequence.sha256").write_text(
        f"{ext_hash}  canonical_multiconcept_sequence.jsonl\n"
        f"{inv_hash}  sequence_generalization_concept_inventory.csv\n",
        encoding="utf-8",
    )

    print(f"Corpus docs: {len(ordered)}  lectures: {n_lectures}")
    print(f"Concepts found in corpus: {len(inv)}   eligible: {len(eligible_concepts)}")
    print(f"Concepts actually used in tasks: {len(concepts_used)}")
    print(f"Tasks: {len(tasks)}  class counts: {labels}")
    print(f"Set sha256: {ext_hash}")
    print()
    print("ELIGIBLE CONCEPT INVENTORY:")
    cols = ["concept", "n_occurrences", "n_lectures_with_occurrence", "first_lecture", "first_slide", "n_slides_before_first"]
    print(eligible_concepts[cols].to_string(index=False))
    print()
    print("EXCLUDED:")
    ex = inv[~inv["eligible"]]
    print(ex[["concept", "n_occurrences", "lecture_coverage_fraction", "first_lecture", "exclusion_reason"]].to_string(index=False))


if __name__ == "__main__":
    main()
