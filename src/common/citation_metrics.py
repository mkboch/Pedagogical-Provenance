def canon(v):
    s = str(v).strip()
    m = re.search(
        r"lecture[_\-\s]*(\d+)[_\-\s]*slide[_\-\s]*(\d+)",
        s,
        flags=re.I
    )
    if not m:
        return s.lower()
    return f"lecture_{int(m.group(1)):02d}_slide_{int(m.group(2)):03d}"

def listish(v):
    if v is None:
        return []
    if isinstance(v, float) and math.isnan(v):
        return []
    if isinstance(v, (list, tuple, set)):
        return [str(x) for x in v]
    s = str(v).strip()
    if not s or s.lower() in {"nan","none","null","[]"}:
        return []
    try:
        x = ast.literal_eval(s)
        if isinstance(x, (list, tuple, set)):
            return [str(z) for z in x]
    except Exception:
        pass
    ids = re.findall(
        r"lecture[_\-\s]*(\d+)[_\-\s]*slide[_\-\s]*(\d+)",
        s,
        flags=re.I
    )
    if ids:
        return [
            f"lecture_{int(a):02d}_slide_{int(b):03d}"
            for a,b in ids
        ]
    return [
        z.strip().strip("'\"")
        for z in re.split(r"[;,|]", s)
        if z.strip()
    ]

CANON = re.compile(
    r"(?<![A-Za-z0-9])"
    r"lecture_(\d+)_slide_(\d+)"
    r"(?![A-Za-z0-9])",
    flags=re.I
)

LOOSE = re.compile(
    r"\bLecture\s*[:#_-]?\s*(\d+)"
    r"\s*[,;/\-]?\s*"
    r"Slide\s*[:#_-]?\s*(\d+)\b",
    flags=re.I
)

def exact_ids(answer):
    return sorted(set(
        f"lecture_{int(a):02d}_slide_{int(b):03d}"
        for a,b in CANON.findall(str(answer))
    ))

def loose_ids(answer):
    return sorted(set(
        f"lecture_{int(a):02d}_slide_{int(b):03d}"
        for a,b in LOOSE.findall(str(answer))
    ))

def citation_metrics(row):
    ans = str(row.get("generated_answer",""))

    exact = set(exact_ids(ans))
    loose = set(loose_ids(ans))

    gold = set(
        canon(x)
        for x in listish(
            row.get("gold_evidence_doc_ids","[]")
        )
    )

    context = set(
        canon(x)
        for x in listish(
            row.get("context_doc_ids","[]")
        )
    )

    exact_real = exact & corpus_ids
    exact_nonexistent = exact - corpus_ids

    loose_real = loose & corpus_ids
    loose_nonexistent = loose - corpus_ids

    # Formal gold citation = canonical citation exactly matching gold.
    gold_formal = exact_real & gold

    # Formal valid citation = canonical citation to any real corpus slide.
    any_valid_formal = exact_real

    # Real but non-gold cited slides.
    real_nongold = exact_real - gold

    # Canonical real slide not in supplied context.
    real_not_context = exact_real - context

    # A "fabricated citation" means it purports to identify a lecture-slide
    # pair that does not exist in the corpus.
    fabricated = exact_nonexistent | loose_nonexistent

    # Real human-readable Lecture X, Slide Y references are NOT fake.
    # They are tracked separately as format variants.
    format_variant_real = loose_real - exact_real

    return {
        "corrected_any_valid_citation":
            int(bool(any_valid_formal)),

        "corrected_gold_citation":
            int(bool(gold_formal)),

        "corrected_fabricated_citation":
            int(bool(fabricated)),

        "corrected_real_nongold_citation":
            int(bool(real_nongold)),

        "corrected_real_not_context_citation":
            int(bool(real_not_context)),

        "real_noncanonical_format_reference":
            int(bool(format_variant_real)),

        "n_exact_canonical_ids":
            len(exact),

        "n_fabricated_ids":
            len(fabricated),

        "exact_ids":
            json.dumps(sorted(exact)),

        "fabricated_ids":
            json.dumps(sorted(fabricated)),
    }
