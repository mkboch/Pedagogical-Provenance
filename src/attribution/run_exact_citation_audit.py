#!/usr/bin/env python3

from pathlib import Path
from collections import defaultdict
import ast
import builtins
import hashlib
import json
import math
import re

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

GEN = (
    ROOT
    / "artifacts"
    / "matched_generation"
)

AGG = (
    GEN
    / "three_model_matched_aggregate"
)

OUT = (
    AGG
    / "citation_evaluation"
)

OUT.mkdir(
    parents=True,
    exist_ok=True,
)


VALIDATED_SOURCE = (
    ROOT
    / "src"
    / "common"
    / "citation_metrics.py"
)

VALIDATED_RESULTS = (
    ROOT
    / "artifacts"
    / "citation_validation"
    / "reference_citation_metrics.csv"
)


CORPUS = (
    ROOT
    / "data"
    / "corpus"
    / "slide_corpus_final.jsonl"
)


MODELS = {
    "qwen3_8b": {
        "display":
            "Qwen3-8B",

        "model_name":
            "Qwen/Qwen3-8B",

        "old":
            ROOT
            / "artifacts/export_packages"
            / "core_evaluation_artifacts"
            / "all_raw_results"
            / "artifacts/generations"
            / "qwen3_8b"
            / "qwen3_8b_full1000_all4systems_4000.jsonl",

        "new":
            GEN
            / "qwen3_8b"
            / "qwen3_8b_matched_position_2000.jsonl",
    },

    "qwen2_5_7b": {
        "display":
            "Qwen2.5-7B-Instruct",

        "model_name":
            "Qwen/Qwen2.5-7B-Instruct",

        "old":
            ROOT
            / "artifacts/export_packages"
            / "core_evaluation_artifacts"
            / "all_raw_results"
            / "artifacts/generations"
            / "qwen2_5_7b"
            / "qwen2_5_7b_full1000_all4systems_4000.jsonl",

        "new":
            GEN
            / "qwen2_5_7b"
            / "qwen2_5_7b_matched_position_2000.jsonl",
    },

    "mistral_7b": {
        "display":
            "Mistral-7B-Instruct-v0.3",

        "model_name":
            "mistralai/Mistral-7B-Instruct-v0.3",

        "old":
            ROOT
            / "artifacts/export_packages"
            / "core_evaluation_artifacts"
            / "all_raw_results"
            / "artifacts/generations"
            / "mistral_7b"
            / "mistral_7b_full1000_all4systems_4000.jsonl",

        "new":
            GEN
            / "mistral_7b"
            / "mistral_7b_matched_position_2000.jsonl",
    },
}


OLD_SYSTEMS = {
    "vanilla_llm",
    "standard_bm25_rag",
    "standard_rrf_bm25_tfidf_rag",
    "slide_indexed_pedagogical_rag",
}


OLD_METHOD = {
    "standard_bm25_rag":
        "bm25",

    "standard_rrf_bm25_tfidf_rag":
        "rrf",
}


ANSWERABLE = {
    "evidence_cited_qa",
    "slide_local_factual_qa",
    "neighbor_slide_conceptual_qa",
}


POLICY = {
    "out_of_scope_abstention_qa",
    "lecture_sequence_violation_probe",
}


EXACT_METRICS = [
    "corrected_any_valid_citation",
    "corrected_gold_citation",
    "corrected_fabricated_citation",
    "corrected_real_nongold_citation",
    "corrected_real_not_context_citation",
    "real_noncanonical_format_reference",
]


DETAIL_METRICS = [
    "n_exact_canonical_ids",
    "n_fabricated_ids",
]


N_BOOT = 20000
N_PERM = 20000

SEED_BOOT = 20260821
SEED_PERM = 20260822


def sha256_file(path):

    h = hashlib.sha256()

    with path.open("rb") as f:

        for block in iter(
            lambda: f.read(
                1024 * 1024
            ),
            b"",
        ):
            h.update(
                block
            )

    return h.hexdigest()


def read_jsonl(path):

    rows = []

    with path.open(
        "r",
        encoding="utf-8",
    ) as f:

        for n, line in enumerate(
            f,
            start=1,
        ):

            if not line.strip():
                continue

            try:

                rows.append(
                    json.loads(
                        line
                    )
                )

            except Exception as e:

                raise RuntimeError(
                    f"Invalid JSON {path}:{n}: {e}"
                )

    return rows


# =============================================================================
# INPUT GATES
# =============================================================================

for p in [
    VALIDATED_SOURCE,
    VALIDATED_RESULTS,
    CORPUS,
]:

    if not p.exists():

        raise SystemExit(
            f"Missing required file: {p}"
        )


for info in MODELS.values():

    for key in [
        "old",
        "new",
    ]:

        if not info[key].exists():

            raise SystemExit(
                f"Missing generation file: "
                f"{info[key]}"
            )


# =============================================================================
# BUILD EXACT CORPUS-ID SET
# =============================================================================

corpus_rows = read_jsonl(
    CORPUS
)


corpus_ids = set()


for row in corpus_rows:

    found = None

    for key in [
        "doc_id",
        "slide_id",
        "id",
        "document_id",
    ]:

        value = row.get(
            key
        )

        if (
            value is not None
            and re.fullmatch(
                r"lecture_\d{2}_slide_\d{3}",
                str(
                    value
                ),
            )
        ):

            found = str(
                value
            )

            break


    if found is None:

        for value in row.values():

            if (
                isinstance(
                    value,
                    str,
                )
                and re.fullmatch(
                    r"lecture_\d{2}_slide_\d{3}",
                    value,
                )
            ):

                found = value

                break


    if found is not None:

        corpus_ids.add(
            found
        )


if len(
    corpus_ids
) != 1117:

    raise SystemExit(
        f"Expected 1117 corpus IDs; "
        f"found {len(corpus_ids)}"
    )


print(
    "CORPUS_ID_GATE = PASSED"
)


# =============================================================================
# PROGRAMMATICALLY EXTRACT EXACT citation_metrics IMPLEMENTATION
#
# We do NOT import src/common/citation_metrics.py because it has executable
# top-level analysis code.
#
# Instead:
#   1. Parse its AST.
#   2. Locate the validated citation_metrics() definition.
#   3. Recursively recover only module-level helper definitions/constants
#      that citation_metrics depends upon and that occur before it.
#   4. Execute those exact AST nodes in an isolated namespace.
#
# This avoids manually recreating any regex or parser logic.
# =============================================================================

source_text = VALIDATED_SOURCE.read_text(
    encoding="utf-8"
)

tree = ast.parse(
    source_text,
    filename=str(
        VALIDATED_SOURCE
    ),
)


target = None


for node in tree.body:

    if (
        isinstance(
            node,
            ast.FunctionDef,
        )
        and node.name
        == "citation_metrics"
    ):

        target = node

        break


if target is None:

    raise SystemExit(
        "Validated citation_metrics() "
        "not found in reference source."
    )


TARGET_LINE = target.lineno


# -------------------------------------------------------------------------
# Index all definitions before/at citation_metrics.
# For duplicate reference helper names, Python would have used the
# latest definition preceding citation_metrics, so we do the same.
# -------------------------------------------------------------------------

definition_nodes = defaultdict(
    list
)


for node in tree.body:

    if node.lineno > TARGET_LINE:

        continue


    if isinstance(
        node,
        ast.FunctionDef,
    ):

        definition_nodes[
            node.name
        ].append(
            node
        )


    elif isinstance(
        node,
        (
            ast.Assign,
            ast.AnnAssign,
        ),
    ):

        names = []


        if isinstance(
            node,
            ast.Assign,
        ):

            targets = node.targets

        else:

            targets = [
                node.target
            ]


        for t in targets:

            if isinstance(
                t,
                ast.Name,
            ):

                names.append(
                    t.id
                )


        for name in names:

            definition_nodes[
                name
            ].append(
                node
            )


def latest_definition(
    name,
):

    nodes = definition_nodes.get(
        name,
        []
    )

    if not nodes:

        return None


    eligible = [
        n
        for n in nodes
        if n.lineno
        <= TARGET_LINE
    ]


    if not eligible:

        return None


    return max(
        eligible,
        key=lambda n:
            n.lineno,
    )


def function_local_names(
    node,
):

    local = set()


    for arg in (
        list(
            node.args.posonlyargs
        )
        + list(
            node.args.args
        )
        + list(
            node.args.kwonlyargs
        )
    ):

        local.add(
            arg.arg
        )


    if node.args.vararg:

        local.add(
            node.args.vararg.arg
        )


    if node.args.kwarg:

        local.add(
            node.args.kwarg.arg
        )


    for n in ast.walk(
        node
    ):

        if (
            isinstance(
                n,
                ast.Name,
            )
            and isinstance(
                n.ctx,
                (
                    ast.Store,
                    ast.Del,
                ),
            )
        ):

            local.add(
                n.id
            )


    return local


def loaded_names(
    node,
):

    names = {
        n.id
        for n in ast.walk(
            node
        )
        if (
            isinstance(
                n,
                ast.Name,
            )
            and isinstance(
                n.ctx,
                ast.Load,
            )
        )
    }


    if isinstance(
        node,
        ast.FunctionDef,
    ):

        names -= function_local_names(
            node
        )


    return names


known_external = (
    set(
        dir(
            builtins
        )
    )
    |
    {
        "re",
        "json",
        "ast",
        "math",
        "np",
        "pd",
        "corpus_ids",
    }
)


needed_names = {
    "citation_metrics"
}

selected = {}

queue = [
    "citation_metrics"
]


unresolved = set()


while queue:

    name = queue.pop()


    if name in selected:

        continue


    node = latest_definition(
        name
    )


    if node is None:

        if name not in known_external:

            unresolved.add(
                name
            )

        continue


    selected[
        name
    ] = node


    for dep in loaded_names(
        node
    ):

        if dep in known_external:

            continue


        dep_node = latest_definition(
            dep
        )


        if dep_node is not None:

            queue.append(
                dep
            )

        else:

            unresolved.add(
                dep
            )


if unresolved:

    raise SystemExit(
        "Unresolved reference parser dependencies: "
        + ", ".join(
            sorted(
                unresolved
            )
        )
    )


# Deduplicate AST nodes because one assignment may define multiple symbols.
unique_nodes = {}


for node in selected.values():

    key = (
        node.lineno,
        getattr(
            node,
            "end_lineno",
            node.lineno,
        ),
        type(
            node
        ).__name__,
    )

    unique_nodes[
        key
    ] = node


nodes_to_exec = sorted(
    unique_nodes.values(),
    key=lambda n:
        n.lineno,
)


parser_source_parts = []


for node in nodes_to_exec:

    segment = ast.get_source_segment(
        source_text,
        node,
    )

    if segment:

        parser_source_parts.append(
            segment
        )


extracted_parser_source = (
    "\n\n".join(
        parser_source_parts
    )
    + "\n"
)


EXTRACTED = (
    OUT
    / "exact_parser_extracted_from_validated_source.py"
)


EXTRACTED.write_text(
    extracted_parser_source,
    encoding="utf-8",
)


namespace = {
    "re":
        re,

    "json":
        json,

    "ast":
        ast,

    "math":
        math,

    "np":
        np,

    "pd":
        pd,

    "corpus_ids":
        corpus_ids,
}


module = ast.Module(
    body=nodes_to_exec,
    type_ignores=[],
)

ast.fix_missing_locations(
    module
)


exec(
    compile(
        module,
        filename=str(
            VALIDATED_SOURCE
        )
        + "::exact_citation_parser",
        mode="exec",
    ),
    namespace,
)


citation_metrics = namespace.get(
    "citation_metrics"
)


if not callable(
    citation_metrics
):

    raise SystemExit(
        "Extracted citation_metrics "
        "is not callable."
    )


print()
print(
    "EXACT_PARSER_EXTRACTION = PASSED"
)

print(
    "Validated source SHA256:",
    sha256_file(
        VALIDATED_SOURCE
    ),
)

print(
    "Extracted parser SHA256:",
    sha256_file(
        EXTRACTED
    ),
)

print(
    "citation_metrics reference line:",
    TARGET_LINE,
)

print(
    "Extracted definition names:"
)

for name in sorted(
    selected
):

    node = selected[
        name
    ]

    print(
        f"  {name}: "
        f"line {node.lineno}"
    )


# =============================================================================
# RECOMPUTE EXACT REFERENCE DIRECT 12,000
# =============================================================================

recomputed_rows = []


for model_key, info in MODELS.items():

    rows = read_jsonl(
        info[
            "old"
        ]
    )


    if len(
        rows
    ) != 4000:

        raise SystemExit(
            f"{model_key}: expected 4000 "
            f"reference direct rows; "
            f"found {len(rows)}"
        )


    for row in rows:

        system = str(
            row.get(
                "system_name",
                "",
            )
        )


        if system not in OLD_SYSTEMS:

            raise SystemExit(
                f"Unexpected old direct system: {system}"
            )


        metrics = citation_metrics(
            row
        )


        out = {
            "model_name":
                str(
                    row[
                        "model_name"
                    ]
                ),

            "system_name":
                system,

            "task_id":
                str(
                    row[
                        "task_id"
                    ]
                ),

            "run_id":
                str(
                    row.get(
                        "run_id",
                        "",
                    )
                ),
        }


        for metric in (
            EXACT_METRICS
            + DETAIL_METRICS
        ):

            out[
                metric
            ] = metrics[
                metric
            ]


        recomputed_rows.append(
            out
        )


recomputed = pd.DataFrame(
    recomputed_rows
)


if len(
    recomputed
) != 12000:

    raise SystemExit(
        f"Expected 12000 reference direct rows; "
        f"found {len(recomputed)}"
    )


# =============================================================================
# LOAD SAVED VALIDATED ROW-LEVEL RESULTS
# =============================================================================

validated = pd.read_csv(
    VALIDATED_RESULTS,
    low_memory=False,
)


required_saved = {
    "model_name",
    "system_name",
    "task_id",
    *EXACT_METRICS,
    *DETAIL_METRICS,
}


missing = (
    required_saved
    -
    set(
        validated.columns
    )
)


if missing:

    raise SystemExit(
        "Saved validated CSV lacks columns: "
        + ", ".join(
            sorted(
                missing
            )
        )
    )


if "source_set" in validated.columns:

    validated = validated.loc[
        validated[
            "source_set"
        ].astype(
            str
        ).eq(
            "original_15k"
        )
    ].copy()


validated = validated.loc[
    validated[
        "system_name"
    ].astype(
        str
    ).isin(
        OLD_SYSTEMS
    )
].copy()


if len(
    validated
) != 12000:

    raise SystemExit(
        f"Expected 12000 saved validated "
        f"original direct rows; "
        f"found {len(validated)}"
    )


KEY = [
    "model_name",
    "system_name",
    "task_id",
]


if recomputed[
    KEY
].duplicated().any():

    raise SystemExit(
        "Recomputed reference keys not unique."
    )


if validated[
    KEY
].duplicated().any():

    raise SystemExit(
        "Saved validated reference keys not unique."
    )


parity = recomputed.merge(
    validated[
        KEY
        + EXACT_METRICS
        + DETAIL_METRICS
    ],
    on=KEY,
    how="outer",
    suffixes=(
        "_recomputed",
        "_validated",
    ),
    indicator=True,
    validate="one_to_one",
)


if len(
    parity
) != 12000:

    raise SystemExit(
        f"Reference parity merge expected "
        f"12000 rows; found {len(parity)}"
    )


if not (
    parity[
        "_merge"
    ]
    == "both"
).all():

    raise SystemExit(
        "Reference row identity parity failed."
    )


parity_failures = []


for metric in (
    EXACT_METRICS
    + DETAIL_METRICS
):

    a = pd.to_numeric(
        parity[
            metric
            + "_recomputed"
        ],
        errors="coerce",
    )


    b = pd.to_numeric(
        parity[
            metric
            + "_validated"
        ],
        errors="coerce",
    )


    mismatch = (
        (
            a.isna()
            != b.isna()
        )
        |
        (
            a.notna()
            & b.notna()
            & (
                a
                != b
            )
        )
    )


    n_bad = int(
        mismatch.sum()
    )


    parity_failures.append(
        {
            "metric":
                metric,

            "N":
                len(
                    parity
                ),

            "mismatches":
                n_bad,

            "passed":
                n_bad
                == 0,
        }
    )


parity_summary = pd.DataFrame(
    parity_failures
)


parity_summary.to_csv(
    OUT
    / "exact_parser_reference_12000_row_parity.csv",
    index=False,
)


print()
print(
    "=" * 120
)

print(
    "EXACT REFERENCE ROW-LEVEL PARITY"
)

print(
    "=" * 120
)

print(
    parity_summary.to_string(
        index=False
    )
)


if not parity_summary[
    "passed"
].all():

    raise SystemExit(
        "EXACT_REFERENCE_CITATION_PARITY = FAILED"
    )


print()
print(
    "EXACT_REFERENCE_CITATION_PARITY = PASSED"
)

print(
    "Rows matched exactly: 12000 / 12000"
)


# =============================================================================
# REFERENCE AGGREGATE SANITY
# =============================================================================

reference_summary = (
    recomputed.groupby(
        "system_name",
        as_index=False,
    )
    .agg(
        N=(
            "task_id",
            "size",
        ),

        corrected_any_valid_citation=(
            "corrected_any_valid_citation",
            "mean",
        ),

        corrected_gold_citation=(
            "corrected_gold_citation",
            "mean",
        ),

        corrected_fabricated_citation=(
            "corrected_fabricated_citation",
            "mean",
        ),

        corrected_real_nongold_citation=(
            "corrected_real_nongold_citation",
            "mean",
        ),

        corrected_real_not_context_citation=(
            "corrected_real_not_context_citation",
            "mean",
        ),

        real_noncanonical_format_reference=(
            "real_noncanonical_format_reference",
            "mean",
        ),
    )
)


reference_summary.to_csv(
    OUT
    / "exact_parser_reference_direct_summary.csv",
    index=False,
)


print()
print(
    "=" * 120
)

print(
    "EXACT REFERENCE CORRECTED-CITATION SUMMARY"
)

print(
    "=" * 120
)

print(
    reference_summary.to_string(
        index=False
    )
)


# =============================================================================
# APPLY SAME EXACT VALIDATED PARSER TO NEW 6,000
# =============================================================================

new_rows = []


for model_key, info in MODELS.items():

    rows = read_jsonl(
        info[
            "new"
        ]
    )


    if len(
        rows
    ) != 2000:

        raise SystemExit(
            f"{model_key}: expected 2000 "
            f"new matched rows; found {len(rows)}"
        )


    for row in rows:

        method = str(
            row.get(
                "retrieval_method",
                "",
            )
        )


        if method not in {
            "bm25",
            "rrf",
        }:

            system = str(
                row.get(
                    "system_name",
                    "",
                )
            )


            if system.startswith(
                "bm25_"
            ):

                method = "bm25"

            elif system.startswith(
                "rrf_"
            ):

                method = "rrf"

            else:

                raise SystemExit(
                    f"Cannot determine matched method: "
                    f"{system}"
                )


        metrics = citation_metrics(
            row
        )


        d = {
            "model_key":
                model_key,

            "model":
                info[
                    "display"
                ],

            "model_name":
                str(
                    row[
                        "model_name"
                    ]
                ),

            "condition":
                "position_matched",

            "method":
                method,

            "task_id":
                str(
                    row[
                        "task_id"
                    ]
                ),

            "task_type":
                str(
                    row[
                        "task_type"
                    ]
                ),

            "system_name":
                str(
                    row[
                        "system_name"
                    ]
                ),
        }


        d.update(
            metrics
        )


        new_rows.append(
            d
        )


new_df = pd.DataFrame(
    new_rows
)


if len(
    new_df
) != 6000:

    raise SystemExit(
        f"Expected 6000 new citation rows; "
        f"found {len(new_df)}"
    )


new_df.to_csv(
    OUT
    / "exact_citation_detail_6000.csv",
    index=False,
)


print()
print(
    "NEW_MATCHED_EXACT_CITATION_ROWS = 6000"
)


# =============================================================================
# CREATE MATCHED REFERENCE GLOBAL BM25/RRF USING EXACT PARSER
# =============================================================================

global_rows = []


for model_key, info in MODELS.items():

    rows = read_jsonl(
        info[
            "old"
        ]
    )


    for row in rows:

        system = str(
            row[
                "system_name"
            ]
        )


        if system not in OLD_METHOD:

            continue


        metrics = citation_metrics(
            row
        )


        d = {
            "model_key":
                model_key,

            "model":
                info[
                    "display"
                ],

            "model_name":
                str(
                    row[
                        "model_name"
                    ]
                ),

            "condition":
                "global_original",

            "method":
                OLD_METHOD[
                    system
                ],

            "task_id":
                str(
                    row[
                        "task_id"
                    ]
                ),

            "task_type":
                str(
                    row[
                        "task_type"
                    ]
                ),

            "system_name":
                system,
        }


        d.update(
            metrics
        )


        global_rows.append(
            d
        )


global_df = pd.DataFrame(
    global_rows
)


if len(
    global_df
) != 6000:

    raise SystemExit(
        f"Expected 6000 original BM25/RRF rows; "
        f"found {len(global_df)}"
    )


combined = pd.concat(
    [
        global_df,
        new_df,
    ],
    ignore_index=True,
)


combined[
    "split"
] = np.where(
    combined[
        "task_type"
    ].isin(
        ANSWERABLE
    ),
    "answerable",
    np.where(
        combined[
            "task_type"
        ].isin(
            POLICY
        ),
        "policy",
        "other",
    ),
)


combined.to_csv(
    OUT
    / "global_vs_position_exact_validated_citation_detail_12000.csv",
    index=False,
)


# =============================================================================
# DESCRIPTIVE SUMMARY
# =============================================================================

summary_parts = []


for split_name, sub in [
    (
        "all",
        combined,
    ),

    (
        "answerable",
        combined.loc[
            combined[
                "split"
            ].eq(
                "answerable"
            )
        ],
    ),

    (
        "policy",
        combined.loc[
            combined[
                "split"
            ].eq(
                "policy"
            )
        ],
    ),
]:

    g = (
        sub.groupby(
            [
                "condition",
                "method",
            ],
            as_index=False,
        )
        .agg(
            N=(
                "task_id",
                "size",
            ),

            **{
                metric: (
                    metric,
                    "mean",
                )
                for metric
                in EXACT_METRICS
            },
        )
    )


    g.insert(
        0,
        "split",
        split_name,
    )


    summary_parts.append(
        g
    )


summary = pd.concat(
    summary_parts,
    ignore_index=True,
)


summary.to_csv(
    OUT
    / "global_vs_position_exact_validated_citation_summary.csv",
    index=False,
)


# =============================================================================
# PAIR GLOBAL VS POSITION
# =============================================================================

g = combined.loc[
    combined[
        "condition"
    ].eq(
        "global_original"
    )
].copy()


p = combined.loc[
    combined[
        "condition"
    ].eq(
        "position_matched"
    )
].copy()


paired = g.merge(
    p,
    on=[
        "model_key",
        "model",
        "task_id",
        "method",
        "task_type",
        "split",
    ],
    how="inner",
    suffixes=(
        "_global",
        "_position",
    ),
    validate="one_to_one",
)


if len(
    paired
) != 6000:

    raise SystemExit(
        f"Expected 6000 paired rows; "
        f"found {len(paired)}"
    )


for metric in EXACT_METRICS:

    paired[
        metric
        + "_delta"
    ] = (
        paired[
            metric
            + "_position"
        ]
        -
        paired[
            metric
            + "_global"
        ]
    )


# =============================================================================
# TASK-CLUSTERED BOOTSTRAP + PAIRED RANDOM-SIGN PERMUTATION
# =============================================================================

def infer(
    df,
    metric,
    seed_offset,
):

    model_counts = (
        df.groupby(
            "task_id"
        )[
            "model_key"
        ]
        .nunique()
    )


    if not (
        model_counts
        == 3
    ).all():

        raise RuntimeError(
            "Not all retained tasks have "
            "three model replicates."
        )


    task_global = (
        df.groupby(
            "task_id"
        )[
            metric
            + "_global"
        ]
        .mean()
    )


    task_position = (
        df.groupby(
            "task_id"
        )[
            metric
            + "_position"
        ]
        .mean()
    )


    delta = (
        task_position
        -
        task_global
    )


    values = delta.to_numpy(
        dtype=float
    )


    n = len(
        values
    )


    observed = float(
        values.mean()
    )


    rng = np.random.default_rng(
        SEED_BOOT
        + seed_offset
    )


    boot = np.empty(
        N_BOOT,
        dtype=float,
    )


    for i in range(
        N_BOOT
    ):

        idx = rng.integers(
            0,
            n,
            size=n,
        )

        boot[
            i
        ] = values[
            idx
        ].mean()


    ci_low = float(
        np.quantile(
            boot,
            0.025,
        )
    )


    ci_high = float(
        np.quantile(
            boot,
            0.975,
        )
    )


    rngp = np.random.default_rng(
        SEED_PERM
        + seed_offset
    )


    extreme = 0

    abs_obs = abs(
        observed
    )


    for _ in range(
        N_PERM
    ):

        signs = rngp.choice(
            np.array(
                [
                    -1.0,
                    1.0,
                ]
            ),
            size=n,
        )


        perm_mean = float(
            np.mean(
                values
                * signs
            )
        )


        if abs(
            perm_mean
        ) >= abs_obs:

            extreme += 1


    p_two = (
        extreme
        + 1
    ) / (
        N_PERM
        + 1
    )


    return {
        "N_unique_tasks":
            n,

        "models_per_task":
            3,

        "global_mean":
            float(
                task_global.mean()
            ),

        "position_mean":
            float(
                task_position.mean()
            ),

        "delta_position_minus_global":
            observed,

        "CI95_low":
            ci_low,

        "CI95_high":
            ci_high,

        "permutation_p_two_sided":
            float(
                p_two
            ),
    }


inference_rows = []

counter = 0


for method in [
    "bm25",
    "rrf",
]:

    md = paired.loc[
        paired[
            "method"
        ].eq(
            method
        )
    ]


    for split_name, sd in [
        (
            "all",
            md,
        ),

        (
            "answerable",
            md.loc[
                md[
                    "split"
                ].eq(
                    "answerable"
                )
            ],
        ),

        (
            "policy",
            md.loc[
                md[
                    "split"
                ].eq(
                    "policy"
                )
            ],
        ),
    ]:


        for metric in EXACT_METRICS:

            counter += 1


            result = infer(
                sd,
                metric,
                counter,
            )


            inference_rows.append(
                {
                    "method":
                        method,

                    "split":
                        split_name,

                    "metric":
                        metric,

                    **result,
                }
            )


inference = pd.DataFrame(
    inference_rows
)


inference.to_csv(
    OUT
    / "exact_validated_citation_task_clustered_inference.csv",
    index=False,
)


# =============================================================================
# PER-MODEL ANSWERABLE EFFECTS
# =============================================================================

per_model_rows = []


for (
    model,
    method,
), md in paired.loc[
    paired[
        "split"
    ].eq(
        "answerable"
    )
].groupby(
    [
        "model",
        "method",
    ]
):

    row = {
        "model":
            model,

        "method":
            method,

        "N":
            len(
                md
            ),
    }


    for metric in EXACT_METRICS:

        row[
            metric
            + "_global"
        ] = float(
            md[
                metric
                + "_global"
            ].mean()
        )

        row[
            metric
            + "_position"
        ] = float(
            md[
                metric
                + "_position"
            ].mean()
        )

        row[
            metric
            + "_delta"
        ] = float(
            md[
                metric
                + "_delta"
            ].mean()
        )


    per_model_rows.append(
        row
    )


per_model = pd.DataFrame(
    per_model_rows
)


per_model.to_csv(
    OUT
    / "exact_validated_citation_per_model_answerable.csv",
    index=False,
)


# =============================================================================
# METADATA
# =============================================================================

metadata = {
    "status":
        "EXACT_VALIDATED_CITATION_COMPLETE",

    "reference_row_parity":
        "12000/12000 exact",

    "validated_source":
        str(
            VALIDATED_SOURCE
        ),

    "validated_source_sha256":
        sha256_file(
            VALIDATED_SOURCE
        ),

    "validated_saved_results":
        str(
            VALIDATED_RESULTS
        ),

    "validated_saved_results_sha256":
        sha256_file(
            VALIDATED_RESULTS
        ),

    "extracted_parser":
        str(
            EXTRACTED
        ),

    "extracted_parser_sha256":
        sha256_file(
            EXTRACTED
        ),

    "reference_citation_function_line":
        TARGET_LINE,

    "new_matched_rows":
        6000,

    "paired_global_position_rows":
        6000,

    "inference_unit":
        "benchmark task",

    "models_per_task":
        3,

    "bootstrap_replicates":
        N_BOOT,

    "permutation_replicates":
        N_PERM,

    "important_note":
        (
            "The citation parser was not manually reconstructed. "
            "Its exact AST definitions were extracted from the "
            "previously validated src/common/citation_metrics.py "
            "source and required to reproduce all saved corrected "
            "citation metrics for all 12,000 reference direct "
            "generations before application to the matched outputs."
        ),
}


(
    OUT
    / "EXACT_VALIDATED_CITATION_METADATA.json"
).write_text(
    json.dumps(
        metadata,
        indent=2,
    ),
    encoding="utf-8",
)


# =============================================================================
# FINAL REPORT
# =============================================================================

print()
print(
    "=" * 120
)

print(
    "NEW MATCHED EXACT-VALIDATED CITATION SUMMARY"
)

print(
    "=" * 120
)

print(
    summary.to_string(
        index=False
    )
)


print()
print(
    "=" * 120
)

print(
    "ANSWERABLE-QA EXACT-VALIDATED CITATION INFERENCE"
)

print(
    "=" * 120
)

print(
    inference.loc[
        inference[
            "split"
        ].eq(
            "answerable"
        )
    ].to_string(
        index=False
    )
)


print()
print(
    "=" * 120
)

print(
    "PER-MODEL ANSWERABLE CITATION EFFECTS"
)

print(
    "=" * 120
)

print(
    per_model.to_string(
        index=False
    )
)


print()
print(
    "EXACT_PARSER_EXTRACTION = PASSED"
)

print(
    "EXACT_REFERENCE_CITATION_PARITY = PASSED"
)

print(
    "EXACT_REFERENCE_ROWS_MATCHED = 12000/12000"
)

print(
    "MATCHED_EXACT_CITATION_ANALYSIS = PASSED"
)

print(
    "NO_GENERATION = TRUE"
)
