"""Prompt and context construction used by the released experiments."""

import re

MAX_CONTEXT_WORDS = 650


def format_context(
    doc_ids,
    corpus_by_doc,
    max_words=MAX_CONTEXT_WORDS,
):
    parts = []
    kept = []
    used = 0

    for d in doc_ids:
        if d not in corpus_by_doc:
            continue

        x = corpus_by_doc[d]

        clean_text = re.sub(
            r"\s+",
            " ",
            x.get("text", ""),
        ).strip()

        chunk = (
            f"[{x['doc_id']} | "
            f"Lecture {x['lecture_id']}, "
            f"Slide {x['slide_id']}]\n"
            f"{clean_text}"
        )

        words = chunk.split()
        remaining = max_words - used

        if remaining <= 0:
            break

        if len(words) > remaining:
            chunk = " ".join(
                words[:remaining]
            ) + " ..."

        parts.append(chunk)
        kept.append(d)

        used += min(
            len(words),
            remaining,
        )

    return "\n\n".join(parts), kept


def make_prompt(
    system_name,
    question,
    context_text="",
):
    if system_name == "vanilla_llm":
        return (
            "You are an educational assistant for a medical imaging course.\n"
            "Answer the student question clearly and concisely.\n\n"
            f"Question:\n{question}\n\n"
            "Answer:"
        )

    if system_name == "standard_bm25_rag":
        return (
            "You are an educational assistant for a medical imaging course.\n"
            "Use only the retrieved lecture evidence below. "
            "If the retrieved evidence is insufficient, say that the lecture "
            "evidence is insufficient. "
            "Do not invent slide IDs or facts.\n\n"
            f"Retrieved evidence:\n{context_text}\n\n"
            f"Question:\n{question}\n\n"
            "Answer:"
        )

    if system_name == "standard_rrf_bm25_tfidf_rag":
        return (
            "You are an educational assistant for a medical imaging course.\n"
            "Use only the retrieved lecture evidence below. "
            "Keep the answer grounded in the evidence. "
            "If the evidence is insufficient, say that the lecture evidence "
            "is insufficient. "
            "Do not invent slide IDs or clinical claims.\n\n"
            f"Retrieved evidence:\n{context_text}\n\n"
            f"Question:\n{question}\n\n"
            "Answer:"
        )

    if system_name == "slide_indexed_pedagogical_rag":
        return (
            "You are a slide-grounded medical imaging teaching assistant. "
            "Answer only from the provided lecture/slide evidence. "
            "Respect the lecture sequence and educational scope. "
            "Do not import later-course concepts unless they are included "
            "in the evidence. "
            "Do not give patient-specific diagnosis, medication dosage, "
            "treatment plans, hospital protocols, or regulatory guidance. "
            "If the slide evidence is insufficient, explicitly say so. "
            "When possible, cite supporting slide IDs.\n\n"
            f"Slide-grounded evidence:\n{context_text}\n\n"
            f"Question:\n{question}\n\n"
            "Answer:"
        )

    raise ValueError(system_name)
