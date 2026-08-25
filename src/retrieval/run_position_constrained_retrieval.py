#!/usr/bin/env python3
from pathlib import Path
import os, re, json, ast, math, argparse, traceback
from collections import Counter, defaultdict
import numpy as np
import pandas as pd

EXCLUDE_DIRS = {
    ".git", "__pycache__", ".ipynb_checkpoints",
    ".conda_experiment", ".venv", "venv", "env",
    "offline_single_package", "offline_analysis_packages",
}

ROOT = None

def norm_col(c):
    return re.sub(r"[^a-z0-9]+", "", str(c).lower())

def pick_col(cols, names):
    nmap = {norm_col(c): c for c in cols}
    for n in names:
        k = norm_col(n)
        if k in nmap:
            return nmap[k]
    return None

def parse_listish(x):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return []
    if isinstance(x, (list, tuple, set)):
        return [str(v) for v in x if str(v).strip()]
    s = str(x).strip()
    if not s or s.lower() in {"nan", "none", "null", "[]"}:
        return []
    try:
        v = ast.literal_eval(s)
        if isinstance(v, (list, tuple, set)):
            return [str(a) for a in v if str(a).strip()]
        if isinstance(v, dict):
            return [str(a) for a in v.values() if str(a).strip()]
    except Exception:
        pass
    parts = re.split(r"[;,|]\s*|\s{2,}", s)
    if len(parts) <= 1:
        parts = re.split(r"\s+", s)
        # Preserve single doc IDs like lecture_14_slide_049.
        if len(parts) > 1 and all("lecture" in p.lower() and "slide" in p.lower() for p in parts):
            return [p.strip() for p in parts if p.strip()]
        if len(parts) > 1:
            return [s]
    return [p.strip().strip("'\"") for p in parts if p.strip().strip("'\"")]

def parse_pos(x):
    if x is None:
        return None
    s = str(x)
    # Common form: lecture_14_slide_049
    pats = [
        r"lecture[_\-\s]*(\d+)[_\-\s]*slide[_\-\s]*(\d+)",
        r"lec(?:ture)?[_\-\s]*(\d+).*?slide[_\-\s]*(\d+)",
        r"l[_\-\s]*(\d+)[_\-\s]*s[_\-\s]*(\d+)",
        r"lecture[_\-\s]*(\d+).*?(\d{1,3})",
    ]
    for pat in pats:
        m = re.search(pat, s.lower())
        if m:
            return (int(m.group(1)), int(m.group(2)))
    return None

def pos_to_aliases(pos):
    if not pos:
        return set()
    l, s = pos
    return {
        f"lecture_{l}_slide_{s}",
        f"lecture_{l:02d}_slide_{s:03d}",
        f"lecture_{l:02d}_slide_{s:02d}",
        f"lecture_{l:02d}_slide_{s}",
        f"lecture_{l}_slide_{s:03d}",
        f"lecture_{l}_slide_{s:02d}",
        f"l{l}_s{s}",
        f"L{l}_S{s}",
    }

def before_or_at(a, b):
    # a <= b in course order
    if a is None or b is None:
        return False
    return (a[0] < b[0]) or (a[0] == b[0] and a[1] <= b[1])

def iter_candidate_files(root):
    for dp, dns, fns in os.walk(root):
        dns[:] = [d for d in dns if d not in EXCLUDE_DIRS and not d.startswith(".conda") and not d.startswith(".venv")]
        for fn in fns:
            if fn.lower().endswith((".csv", ".jsonl", ".json")):
                p = Path(dp) / fn
                try:
                    if p.stat().st_size > 800_000_000:
                        continue
                except Exception:
                    continue
                yield p

def read_sample(p, n=100):
    try:
        if p.suffix.lower() == ".csv":
            return pd.read_csv(p, nrows=n)
        if p.suffix.lower() == ".jsonl":
            rows = []
            with p.open("r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rows.append(json.loads(line))
                    except Exception:
                        continue
                    if len(rows) >= n:
                        break
            return pd.DataFrame(rows)
        if p.suffix.lower() == ".json":
            obj = json.loads(p.read_text(encoding="utf-8", errors="ignore"))
            if isinstance(obj, list):
                return pd.DataFrame(obj[:n])
            if isinstance(obj, dict):
                for v in obj.values():
                    if isinstance(v, list) and v and isinstance(v[0], dict):
                        return pd.DataFrame(v[:n])
                return pd.DataFrame([obj])
    except Exception:
        return None
    return None

def read_full(p):
    if p.suffix.lower() == ".csv":
        return pd.read_csv(p)
    if p.suffix.lower() == ".jsonl":
        rows = []
        with p.open("r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        rows.append(json.loads(line))
                    except Exception:
                        pass
        return pd.DataFrame(rows)
    if p.suffix.lower() == ".json":
        obj = json.loads(p.read_text(encoding="utf-8", errors="ignore"))
        if isinstance(obj, list):
            return pd.DataFrame(obj)
        if isinstance(obj, dict):
            for v in obj.values():
                if isinstance(v, list) and v and isinstance(v[0], dict):
                    return pd.DataFrame(v)
            return pd.DataFrame([obj])
    raise ValueError(f"Unsupported file: {p}")

def find_task_file(root):
    best = []
    for p in iter_candidate_files(root):
        df = read_sample(p)
        if df is None or df.empty:
            continue
        cols = list(df.columns)
        task_col = pick_col(cols, ["task_id", "Task_ID", "qid", "question_id", "id"])
        q_col = pick_col(cols, ["question", "Question", "query", "prompt"])
        type_col = pick_col(cols, ["task_type", "Task_Type", "task_family", "type"])
        target_col = pick_col(cols, ["target_doc_id", "Target_Doc_ID", "target_slide_id", "target_id", "target_slide", "doc_id"])
        gold_col = pick_col(cols, [
            "gold_evidence_doc_ids", "Gold_Evidence_Doc_IDs", "evidence_doc_ids",
            "Evidence_Slide_IDs", "evidence_slide_ids", "gold_ids", "evidence_ids",
            "reference_evidence_ids", "gold_context_doc_ids"
        ])
        score = 0
        if task_col: score += 2
        if q_col: score += 4
        if type_col: score += 1
        if target_col: score += 3
        if gold_col: score += 4
        name = str(p).lower()
        if "human_eval" in name and "unblinded" in name:
            score -= 4
        if "full1000" in name or "benchmark" in name:
            score += 3
        if "detail" in name:
            score += 1
        if score >= 8:
            best.append((score, p, cols, task_col, q_col, type_col, target_col, gold_col))
    if not best:
        raise RuntimeError("Could not auto-detect a benchmark task file with task_id/question/target/gold evidence.")
    best.sort(key=lambda x: (x[0], x[1].stat().st_size), reverse=True)
    return best[0], best[:10]

def find_corpus_file(root):
    preferred = root / "artifacts/retrieval_inputs/discovered_slide_corpus.csv"
    candidates = []
    paths = [preferred] if preferred.exists() else []
    paths += list(iter_candidate_files(root))
    seen = set()
    for p in paths:
        if p in seen:
            continue
        seen.add(p)
        df = read_sample(p, n=200)
        if df is None or df.empty:
            continue
        cols = list(df.columns)
        id_col = pick_col(cols, ["doc_id", "slide_id", "lecture_slide_id", "slide_key", "key", "id", "Target_Doc_ID"])
        text_col = pick_col(cols, ["text", "transcript", "content", "slide_text", "raw_text", "page_content", "Evidence_Text"])
        lec_col = pick_col(cols, ["lecture", "lecture_id", "lecture_idx", "lecture_index", "lecture_no", "Lecture"])
        slide_col = pick_col(cols, ["slide", "slide_id", "slide_idx", "slide_index", "slide_no", "Slide"])
        score = 0
        if id_col: score += 3
        if text_col: score += 5
        if lec_col and slide_col: score += 3
        name = str(p).lower()
        if "discovered_slide_corpus" in name:
            score += 10
        if "corpus" in name:
            score += 3
        if score >= 7:
            candidates.append((score, p, cols, id_col, text_col, lec_col, slide_col))
    if not candidates:
        raise RuntimeError("Could not auto-detect slide corpus file.")
    candidates.sort(key=lambda x: (x[0], x[1].stat().st_size), reverse=True)
    return candidates[0], candidates[:10]

def normalize_tasks(df, cols_meta):
    _, p, cols, task_col, q_col, type_col, target_col, gold_col = cols_meta
    out = []
    for _, r in df.iterrows():
        tid = str(r.get(task_col, "")).strip()
        q = str(r.get(q_col, "")).strip()
        if not tid or not q or q.lower() == "nan":
            continue
        task_type = str(r.get(type_col, "")).strip() if type_col else ""
        target_raw = str(r.get(target_col, "")).strip() if target_col else ""
        gold_raw = r.get(gold_col, None) if gold_col else None
        gold_ids = parse_listish(gold_raw)
        if not gold_ids and target_raw:
            gold_ids = [target_raw]
        target_pos = parse_pos(target_raw)
        if target_pos is None and gold_ids:
            target_pos = parse_pos(gold_ids[0])
        gold_pos = [parse_pos(x) for x in gold_ids]
        gold_pos = [x for x in gold_pos if x is not None]
        if not target_pos or not gold_pos:
            continue
        out.append({
            "task_id": tid,
            "task_type": task_type,
            "question": q,
            "target_raw": target_raw,
            "target_pos": target_pos,
            "gold_ids_raw": gold_ids,
            "gold_pos": gold_pos,
        })
    # Deduplicate by task_id.
    dedup = {}
    for t in out:
        dedup.setdefault(t["task_id"], t)
    return list(dedup.values())

def normalize_corpus(df, cols_meta):
    _, p, cols, id_col, text_col, lec_col, slide_col = cols_meta
    docs = []
    for _, r in df.iterrows():
        text = str(r.get(text_col, "")).strip() if text_col else ""
        if not text or text.lower() == "nan":
            continue
        raw_id = str(r.get(id_col, "")).strip() if id_col else ""
        pos = parse_pos(raw_id)
        if pos is None and lec_col and slide_col:
            try:
                pos = (int(float(r.get(lec_col))), int(float(r.get(slide_col))))
            except Exception:
                pos = None
        if not raw_id and pos:
            raw_id = f"lecture_{pos[0]:02d}_slide_{pos[1]:03d}"
        if pos is None:
            continue
        docs.append({
            "doc_id": raw_id,
            "pos": pos,
            "lecture": pos[0],
            "slide": pos[1],
            "text": text,
        })
    # Deduplicate by position, keep longest text.
    bypos = {}
    for d in docs:
        if d["pos"] not in bypos or len(d["text"]) > len(bypos[d["pos"]]["text"]):
            bypos[d["pos"]] = d
    return list(bypos.values())

def tokenize(s):
    return re.findall(r"[a-z0-9]+", str(s).lower())

class BM25:
    def __init__(self, tokenized_docs, k1=1.5, b=0.75):
        self.docs = tokenized_docs
        self.N = len(tokenized_docs)
        self.avgdl = sum(len(d) for d in tokenized_docs) / max(1, self.N)
        self.k1 = k1
        self.b = b
        self.tf = []
        df = Counter()
        for d in tokenized_docs:
            c = Counter(d)
            self.tf.append(c)
            for term in c:
                df[term] += 1
        self.idf = {t: math.log(1 + (self.N - n + 0.5) / (n + 0.5)) for t, n in df.items()}
    def score_one_doc(self, qterms, i):
        dl = len(self.docs[i])
        denom_base = self.k1 * (1 - self.b + self.b * dl / max(1e-9, self.avgdl))
        c = self.tf[i]
        score = 0.0
        for t in qterms:
            f = c.get(t, 0)
            if f:
                score += self.idf.get(t, 0.0) * (f * (self.k1 + 1)) / (f + denom_base)
        return score
    def scores(self, qterms, indices):
        return np.array([self.score_one_doc(qterms, i) for i in indices], dtype=float)

def topk_from_scores(indices, scores, k):
    if len(indices) == 0:
        return []
    order = np.argsort(-scores)[:k]
    return [indices[int(i)] for i in order]

def eval_retrieved(task, retrieved_docs):
    retrieved_pos = [d["pos"] for d in retrieved_docs]
    retrieved_set = set(retrieved_pos)
    gold_set = set(task["gold_pos"])
    target_pos = task["target_pos"]
    return {
        "any_gold": int(bool(retrieved_set & gold_set)),
        "all_gold": int(gold_set.issubset(retrieved_set)),
        "target_slide": int(target_pos in retrieved_set),
        "target_lecture": int(any(p[0] == target_pos[0] for p in retrieved_pos)),
        "ctx": len(retrieved_docs),
        "retrieved_ids": json.dumps([d["doc_id"] for d in retrieved_docs]),
        "retrieved_positions": json.dumps([list(d["pos"]) for d in retrieved_docs]),
    }

def summarize(detail):
    df = pd.DataFrame(detail)
    rows = []
    for method, g in df.groupby("method"):
        rows.append({
            "method": method,
            "N": len(g),
            "Ctx": g["ctx"].mean(),
            "Any gold": g["any_gold"].mean(),
            "All gold": g["all_gold"].mean(),
            "Target slide": g["target_slide"].mean(),
            "Target lecture": g["target_lecture"].mean(),
        })
    return pd.DataFrame(rows).sort_values("method")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(Path(__file__).resolve().parents[2]))
    ap.add_argument("--out", default="artifacts/position_constrained_retrieval")
    ap.add_argument("--k", type=int, default=3)
    args = ap.parse_args()

    root = Path(args.root).expanduser().resolve()
    out = (root / args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)

    print("ROOT:", root)
    print("OUT :", out)
    print("K   :", args.k)

    task_best, task_candidates = find_task_file(root)
    corpus_best, corpus_candidates = find_corpus_file(root)

    print("\n===== SELECTED TASK FILE =====")
    print(task_best[1])
    print("score:", task_best[0])
    print("columns:", task_best[2])
    print("task_col/question/type/target/gold:", task_best[3:])

    print("\n===== SELECTED CORPUS FILE =====")
    print(corpus_best[1])
    print("score:", corpus_best[0])
    print("columns:", corpus_best[2])
    print("id/text/lecture/slide:", corpus_best[3:])

    tasks_df = read_full(task_best[1])
    corpus_df = read_full(corpus_best[1])

    tasks = normalize_tasks(tasks_df, task_best)
    docs = normalize_corpus(corpus_df, corpus_best)

    if len(tasks) < 500:
        print("\nWARNING: fewer than 500 tasks found. Top task candidates were:")
        for c in task_candidates:
            print(" ", c[0], c[1])
    if len(docs) < 500:
        print("\nWARNING: fewer than 500 docs found. Top corpus candidates were:")
        for c in corpus_candidates:
            print(" ", c[0], c[1])

    print("\n===== NORMALIZED DATA =====")
    print("tasks:", len(tasks))
    print("docs :", len(docs))
    print("task types:", Counter(t["task_type"] for t in tasks))

    if len(tasks) == 0 or len(docs) == 0:
        raise RuntimeError("No tasks or docs after normalization.")

    # Sort docs in course order.
    docs = sorted(docs, key=lambda d: (d["lecture"], d["slide"], d["doc_id"]))
    texts = [d["text"] for d in docs]
    doc_tokens = [tokenize(x) for x in texts]
    q_tokens = [tokenize(t["question"]) for t in tasks]

    print("\n===== BUILD BM25 =====")
    bm25 = BM25(doc_tokens)

    print("===== BUILD TF-IDF FOR RRF =====")
    from sklearn.feature_extraction.text import TfidfVectorizer
    vectorizer = TfidfVectorizer(lowercase=True, token_pattern=r"(?u)\b[a-zA-Z0-9]+\b")
    X = vectorizer.fit_transform(texts)
    Q = vectorizer.transform([t["question"] for t in tasks])

    def candidate_indices(task, constrained):
        if not constrained:
            return list(range(len(docs)))
        inds = [i for i, d in enumerate(docs) if before_or_at(d["pos"], task["target_pos"])]
        if not inds:
            return list(range(len(docs)))
        return inds

    detail = []

    print("\n===== RUN BM25 + RRF GLOBAL/POSITION-CONSTRAINED =====")
    for ti, task in enumerate(tasks):
        if ti % 100 == 0:
            print(f"retrieval progress {ti}/{len(tasks)}")
        for constrained, suffix in [(False, "global"), (True, "position_constrained")]:
            inds = candidate_indices(task, constrained)

            # BM25
            s_bm25 = bm25.scores(q_tokens[ti], inds)
            top_bm25 = topk_from_scores(inds, s_bm25, args.k)
            row = {
                "task_id": task["task_id"],
                "task_type": task["task_type"],
                "method": f"bm25_{suffix}",
                "target_pos": str(task["target_pos"]),
                "gold_pos": json.dumps([list(x) for x in task["gold_pos"]]),
                "candidate_count": len(inds),
            }
            row.update(eval_retrieved(task, [docs[i] for i in top_bm25]))
            detail.append(row)

            # RRF = BM25 rank + TFIDF rank within same candidate set.
            s_tfidf = np.asarray((Q[ti] @ X[inds].T).todense()).ravel()
            bm_order = np.argsort(-s_bm25)
            tf_order = np.argsort(-s_tfidf)
            rrf_scores = defaultdict(float)
            for rank, oi in enumerate(bm_order, start=1):
                rrf_scores[inds[int(oi)]] += 1.0 / (60.0 + rank)
            for rank, oi in enumerate(tf_order, start=1):
                rrf_scores[inds[int(oi)]] += 1.0 / (60.0 + rank)
            top_rrf = [i for i, _ in sorted(rrf_scores.items(), key=lambda kv: -kv[1])[:args.k]]
            row = {
                "task_id": task["task_id"],
                "task_type": task["task_type"],
                "method": f"rrf_{suffix}",
                "target_pos": str(task["target_pos"]),
                "gold_pos": json.dumps([list(x) for x in task["gold_pos"]]),
                "candidate_count": len(inds),
            }
            row.update(eval_retrieved(task, [docs[i] for i in top_rrf]))
            detail.append(row)

    print("\n===== RUN BGE GLOBAL/POSITION-CONSTRAINED =====")
    try:
        import torch
        from sentence_transformers import SentenceTransformer
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print("BGE device:", device)
        model = SentenceTransformer("BAAI/bge-base-en-v1.5", device=device)

        doc_emb = model.encode(
            texts,
            batch_size=64,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=True,
        )
        queries = ["Represent this sentence for searching relevant passages: " + t["question"] for t in tasks]
        q_emb = model.encode(
            queries,
            batch_size=64,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=True,
        )

        for ti, task in enumerate(tasks):
            if ti % 100 == 0:
                print(f"BGE progress {ti}/{len(tasks)}")
            for constrained, suffix in [(False, "global"), (True, "position_constrained")]:
                inds = candidate_indices(task, constrained)
                sims = doc_emb[inds] @ q_emb[ti]
                top = topk_from_scores(inds, sims, args.k)
                row = {
                    "task_id": task["task_id"],
                    "task_type": task["task_type"],
                    "method": f"bge_{suffix}",
                    "target_pos": str(task["target_pos"]),
                    "gold_pos": json.dumps([list(x) for x in task["gold_pos"]]),
                    "candidate_count": len(inds),
                }
                row.update(eval_retrieved(task, [docs[i] for i in top]))
                detail.append(row)
    except Exception as e:
        print("BGE FAILED. BM25/RRF results still saved.")
        print(type(e).__name__, str(e))
        traceback.print_exc()

    detail_df = pd.DataFrame(detail)
    detail_path = out / "position_constrained_retrieval_detail.csv"
    detail_df.to_csv(detail_path, index=False)

    summary = summarize(detail_df)
    summary_path = out / "table_position_constrained_retrieval_summary.csv"
    summary.to_csv(summary_path, index=False)

    by_task = detail_df.groupby(["task_type", "method"]).agg(
        N=("task_id", "count"),
        Ctx=("ctx", "mean"),
        Any_gold=("any_gold", "mean"),
        All_gold=("all_gold", "mean"),
        Target_slide=("target_slide", "mean"),
        Target_lecture=("target_lecture", "mean"),
        Candidate_count=("candidate_count", "mean"),
    ).reset_index()
    by_task_path = out / "table_position_constrained_retrieval_by_task.csv"
    by_task.to_csv(by_task_path, index=False)

    meta = {
        "task_file": str(task_best[1]),
        "corpus_file": str(corpus_best[1]),
        "n_tasks": len(tasks),
        "n_docs": len(docs),
        "k": args.k,
        "methods": sorted(detail_df["method"].unique().tolist()),
        "artifacts": {
            "detail": str(detail_path),
            "summary": str(summary_path),
            "by_task": str(by_task_path),
        },
    }
    (out / "position_constrained_retrieval_metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print("\n===== POSITION-CONSTRAINED RETRIEVAL SUMMARY =====")
    print(summary.to_string(index=False))

    print("\n===== BY TASK SUMMARY =====")
    print(by_task.to_string(index=False))

    print("\nOUTPUTS:")
    print(detail_path)
    print(summary_path)
    print(by_task_path)
    print(out / "position_constrained_retrieval_metadata.json")
    print("\nPOSITION CONSTRAINED RETRIEVAL OK")

if __name__ == "__main__":
    main()
