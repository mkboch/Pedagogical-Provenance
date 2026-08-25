#!/usr/bin/env python3
"""RRF-lecture -> slide two-stage hybrid, position-constrained protocol.

STAGE 1: rank LECTURES using aggregated RRF slide evidence.
STAGE 2: rank slides within the selected lecture budget L, by RRF or BGE.

RETRIEVAL ONLY. No generation. CPU only. No gold information is used at any
ranking stage.
"""
from pathlib import Path
import ast, json, re
import numpy as np, pandas as pd, torch

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/"artifacts/sequence_policy"
CORPUS = ROOT/"data/corpus/slide_corpus_final.jsonl"
BENCH = ROOT/"data/benchmark/core_qa_benchmark.jsonl"
REFERENCE = ROOT / "src/common/runtime_reference.py"
ANS = {"evidence_cited_qa","slide_local_factual_qa","neighbor_slide_conceptual_qa"}
KS = [1,3,5,10,20,50]
LBUDGETS = [1,3,5]

def log(*a): print(*a, flush=True)

# reference scorers, verbatim
from collections import defaultdict
_t=ast.parse(REFERENCE.read_text(encoding="utf-8"))
_want={"norm","tokenize","unique","doc_text","simple_bm25_scores","rrf"}
_b=[n for n in _t.body if isinstance(n,(ast.Assign,ast.AnnAssign)) and (lambda x: True)(n)]
_b=[n for n in _b if (lambda n: (lambda: True)())(n)]
_keep=[]
for n in _t.body:
    if isinstance(n,(ast.Assign,ast.AnnAssign)):
        try: ast.literal_eval(n.value); _keep.append(n)
        except Exception: pass
_keep+=[n for n in _t.body if isinstance(n,ast.FunctionDef) and n.name in _want]
_m=ast.Module(body=_keep,type_ignores=[]); ast.fix_missing_locations(_m)
ns={"re":re,"np":np,"json":json,"math":__import__("math"),"defaultdict":defaultdict}
exec(compile(_m,str(REFERENCE),"exec"),ns,ns)
tokenize,doc_text,bm25_fn,rrf_fn = ns["tokenize"],ns["doc_text"],ns["simple_bm25_scores"],ns["rrf"]
log("recovered reference helpers verbatim")

corpus=[json.loads(l) for l in CORPUS.read_text(encoding="utf-8").splitlines() if l.strip()]
assert len(corpus)==1117
for d in corpus: d["lecture_id"]=int(d["lecture_id"]); d["slide_id"]=int(d["slide_id"])
ordered=sorted(corpus,key=lambda d:(d["lecture_id"],d["slide_id"]))
for i,d in enumerate(ordered): d["cp"]=i
doc_ids=[d["doc_id"] for d in ordered]
lec=[d["lecture_id"] for d in ordered]
pos_of={d["doc_id"]:d["cp"] for d in ordered}
lec_of={d["doc_id"]:d["lecture_id"] for d in ordered}

bench=pd.read_json(BENCH,lines=True); bench=bench[bench.task_type.isin(ANS)]
tasks=[]
for _,r in bench.iterrows():
    t=str(r["target_doc_id"])
    if t in pos_of:
        tasks.append({"task_id":str(r["task_id"]),"task_type":r["task_type"],
            "q":r["question"],"target":t,"gold":set(str(x) for x in r["evidence_doc_ids"]),
            "tp":pos_of[t],"tl":lec_of[t]})
tasks=sorted(tasks,key=lambda x:x["task_id"]); assert len(tasks)==600
log(f"answerable tasks={len(tasks)}")

doc_tok=[tokenize(doc_text(d)) for d in ordered]
from sklearn.feature_extraction.text import TfidfVectorizer
tfv=TfidfVectorizer(lowercase=True,stop_words="english",ngram_range=(1,3),max_df=.85)
tfm=tfv.fit_transform([doc_text(d) for d in ordered])
from sentence_transformers import SentenceTransformer
if torch.cuda.is_available(): raise SystemExit("SAFETY STOP: CUDA")
em=SentenceTransformer("BAAI/bge-base-en-v1.5",device="cpu")
demb=em.encode([d.get("text","") for d in ordered],batch_size=32,convert_to_numpy=True,
               normalize_embeddings=True,show_progress_bar=True)
qemb=em.encode([t["q"] for t in tasks],batch_size=32,convert_to_numpy=True,normalize_embeddings=True)
log("embeddings ready")

rank_store={}
for qi,t in enumerate(tasks):
    es={i for i,d in enumerate(ordered) if d["cp"]<=t["tp"]}
    qt=tokenize(t["q"]); bs=bm25_fn(qt,doc_tok)
    ts=(tfm@tfv.transform([t["q"]]).T).toarray().ravel()
    ds=demb@qemb[qi]
    def top(s,k=50):
        o=np.argsort(-np.asarray(s),kind="stable")
        return [int(i) for i in o if i in es][:k]
    rb,rt,rg=top(bs),top(ts),top(ds)
    rrf_full=[i for i in rrf_fn([rb,rt]) if i in es]
    # STAGE 1: lecture scores = sum of RRF-style reciprocal rank over that lecture's slides
    lscore=defaultdict(float)
    for rank,i in enumerate(rrf_full,1): lscore[lec[i]] += 1.0/(60+rank)
    lec_rank=[L for L,_ in sorted(lscore.items(), key=lambda kv:(-kv[1],kv[0]))]
    rank_store[t["task_id"]]={"rrf_full":rrf_full,"lec_rank":lec_rank,
                              "dense":[i for i in rg],"es":es,
                              "dense_scores":ds}
    if (qi+1)%100==0: log(f"  stage1 {qi+1}/600")

def build(t,L,slide_scorer):
    st=rank_store[t["task_id"]]
    keep=set(st["lec_rank"][:L])
    cand=[i for i in st["rrf_full"] if lec[i] in keep] if slide_scorer=="rrf" else \
         [i for i in np.argsort(-st["dense_scores"],kind="stable") if i in st["es"] and lec[i] in keep]
    # backfill beyond lecture budget so deeper k remains measurable
    rest=[i for i in st["rrf_full"] if i not in cand]
    return [doc_ids[i] for i in list(dict.fromkeys(list(cand)+rest))]

rows=[];mrr=[]
methods={}
for L in LBUDGETS:
    for sc in ["rrf","bge"]:
        methods[f"rrf_lec_L{L}_slide_{sc}"]=lambda t,L=L,sc=sc: build(t,L,sc)
for name,fn in methods.items():
    for t in tasks:
        r=fn(t)
        fg=next((i+1 for i,d in enumerate(r) if d in t["gold"]),None)
        ft=next((i+1 for i,d in enumerate(r) if d==t["target"]),None)
        mrr.append({"method":name,"task_id":t["task_id"],"task_type":t["task_type"],
                    "first_gold_rank":fg,"first_target_rank":ft,
                    "rr_any_gold":1/fg if fg else 0.,"rr_target":1/ft if ft else 0.})
        for k in KS:
            top=set(r[:k])
            rows.append({"method":name,"k":k,"task_id":t["task_id"],"task_type":t["task_type"],
                "any_gold":int(bool(top&t["gold"])),
                "all_gold":int(bool(t["gold"]) and t["gold"]<=top),
                "target_slide":int(t["target"] in top),
                "target_lecture":int(any(lec_of[d]==t["tl"] for d in r[:k]))})
    log(f"evaluated {name}")

K=pd.DataFrame(rows);M=pd.DataFrame(mrr)
# authoritative RRF/BM25 for comparison
prev=pd.read_csv(ROOT/"artifacts/retrieval_diagnostics/advanced_retrieval_k_sweep_detail.csv")
prev=prev[prev.method.isin(["bm25","rrf","bge_dense","hier_bge_L1","bge_reranker_union50"])]
K=pd.concat([K,prev[["method","k","task_id","task_type","any_gold","all_gold","target_slide","target_lecture"]]],ignore_index=True)
pm=pd.read_csv(ROOT/"artifacts/retrieval_diagnostics/advanced_retrieval_mrr_detail.csv")
pm=pm[pm.method.isin(["bm25","rrf","bge_dense","hier_bge_L1","bge_reranker_union50"])]
M=pd.concat([M,pm[["method","task_id","task_type","first_gold_rank","first_target_rank","rr_any_gold","rr_target"]]],ignore_index=True)

sw=(K.groupby(["method","k"],as_index=False).agg(N=("task_id","nunique"),
    AnyGoldRecall=("any_gold","mean"),AllGoldRecall=("all_gold","mean"),
    TargetSlideRecall=("target_slide","mean"),TargetLectureRecall=("target_lecture","mean")))
sw.to_csv(OUT/"rrf_hierarchical_k_sweep.csv",index=False)
mr=(M.groupby("method",as_index=False).agg(N=("task_id","nunique"),
    MRRAnyGold=("rr_any_gold","mean"),MRRTargetSlide=("rr_target","mean"),
    MedianFirstGoldRank=("first_gold_rank","median"),MedianFirstTargetRank=("first_target_rank","median")))
mr.to_csv(OUT/"rrf_hierarchical_mrr.csv",index=False)
log("\n"+sw[sw.k.isin([1,3,10,50])].to_string(index=False))
log("\n"+mr.to_string(index=False))
log("RRF_LECTURE_SLIDE_HYBRID = DONE (retrieval only, no generation)")
