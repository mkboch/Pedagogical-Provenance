"""Retrieval primitives used by the released experiments.

These functions are reproduced from the scoring implementation used for
the reported benchmark experiments so that the public ranker does not
depend on an archived project script.
"""

import json
import math
import re
from collections import defaultdict

import numpy as np

TOP_K = 3
MAX_CONTEXT_WORDS = 650
MAX_NEW_TOKENS = 256
MODELS = [{'name': 'Qwen/Qwen3-8B', 'tag': 'qwen3_8b', 'cache_glob': 'models--Qwen--Qwen3-8B'}, {'name': 'Qwen/Qwen2.5-7B-Instruct', 'tag': 'qwen2_5_7b', 'cache_glob': 'models--Qwen--Qwen2.5-7B-Instruct'}, {'name': 'mistralai/Mistral-7B-Instruct-v0.3', 'tag': 'mistral_7b', 'cache_glob': 'models--mistralai--Mistral-7B-Instruct-v0.3'}]
SYSTEM_ORDER = ['vanilla_llm', 'standard_bm25_rag', 'standard_rrf_bm25_tfidf_rag', 'slide_indexed_pedagogical_rag', 'slide_indexed_pedagogical_rag_plus_verifier']
SYSTEM_NAMES = {'vanilla_llm': 'Vanilla LLM', 'standard_bm25_rag': 'BM25 RAG', 'standard_rrf_bm25_tfidf_rag': 'RRF RAG', 'slide_indexed_pedagogical_rag': 'Slide-indexed RAG', 'slide_indexed_pedagogical_rag_plus_verifier': 'Slide-indexed RAG + verifier'}

def norm(s):
    return re.sub('\\s+', ' ', (s or '').lower()).strip()

def tokenize(text):
    return re.findall('[a-zA-Z][a-zA-Z0-9\\-]+', norm(text))

def unique(seq):
    seen = set()
    out = []
    for x in seq:
        if x and x not in seen:
            seen.add(x)
            out.append(x)
    return out

def doc_text(x):
    return f"Lecture {x['lecture_id']}, Slide {x['slide_id']}. Document ID: {x['doc_id']}. {x.get('text', '')}"

def simple_bm25_scores(query_tokens, doc_tokens):
    N = len(doc_tokens)
    avgdl = sum((len(d) for d in doc_tokens)) / max(1, N)
    df = defaultdict(int)
    for d in doc_tokens:
        for t in set(d):
            df[t] += 1
    idf = {t: np.log(1 + (N - n + 0.5) / (n + 0.5)) for t, n in df.items()}
    k1 = 1.5
    b = 0.75
    q_terms = list(dict.fromkeys(query_tokens))
    scores = []
    for d in doc_tokens:
        tf = defaultdict(int)
        for t in d:
            tf[t] += 1
        dl = len(d)
        s = 0.0
        for t in q_terms:
            if t not in tf:
                continue
            denom = tf[t] + k1 * (1 - b + b * dl / avgdl)
            s += idf.get(t, 0.0) * (tf[t] * (k1 + 1)) / denom
        scores.append(s)
    return np.array(scores)

def rrf(rank_lists, k=60):
    score = defaultdict(float)
    for ranks in rank_lists:
        for idx, d in enumerate(ranks, start=1):
            score[d] += 1.0 / (k + idx)
    return [d for d, _ in sorted(score.items(), key=lambda x: x[1], reverse=True)]


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
