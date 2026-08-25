#!/usr/bin/env python3
"""stronger cross-encoder reranker under the position-constrained protocol.

RETRIEVAL ONLY. No answer generation. CPU only.

Model: BAAI/bge-reranker-v2-m3 (larger XLM-R-based reranker) vs the
BAAI/bge-reranker-base result already measured in the advanced-retrieval
run.

PRESPECIFIED SCOPE (fixed before any result was inspected): because the
larger model is ~2x the parameters and this must run on CPU, evaluation
uses a DETERMINISTIC every-3rd-task subsample of the 600 answerable tasks
(n=200). The already-computed bge-reranker-base results are subset to the
SAME 200 tasks so the comparison is matched. Two passage policies are run:
  A. truncation-at-512 (identical to the base-reranker baseline)
  B. chunked passage scoring (max-over-chunks), for long transcripts
"""
from pathlib import Path
import ast, json, math, re, time
import numpy as np, pandas as pd, torch

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/"artifacts/final_targeted_validation_20260822"
PROTOCOL_REFERENCE = ROOT/"artifacts/matched_retrieval/run_matched_retrieval_core.py"
ADV = ROOT/"src/retrieval/run_advanced_retrieval.py"
CORPUS = ROOT/"data/corpus/slide_corpus_final.jsonl"
BENCH = ROOT/"data/benchmark/core_qa_benchmark.jsonl"

MODEL = "BAAI/bge-reranker-v2-m3"
BASE_MODEL = "BAAI/bge-reranker-base"
BATCH = 16
MAXLEN = 512
CHUNK_WORDS, CHUNK_STRIDE = 180, 150
SUBSAMPLE_STRIDE = 1  # FULL 600 (analysis stage)
ANS = {"evidence_cited_qa","slide_local_factual_qa","neighbor_slide_conceptual_qa"}
KS = [1,3,5,10,20,50]

def log(*a):
    print(*a, flush=True)

# Recover the REFERENCE scoring helpers verbatim, exactly as both the reference
# freeze and the advanced-retrieval script do (they are defined in the
# reference pipeline, not at those scripts' top level).
REFERENCE = ROOT / "src/common/runtime_reference.py"
from collections import defaultdict
_tree = ast.parse(REFERENCE.read_text(encoding="utf-8"))
_want = {"norm","tokenize","unique","doc_text","simple_bm25_scores","rrf"}
_body = []
for n in _tree.body:
    if isinstance(n,(ast.Assign,ast.AnnAssign)):
        try: ast.literal_eval(n.value); _body.append(n)
        except Exception: pass
_body += [n for n in _tree.body if isinstance(n,ast.FunctionDef) and n.name in _want]
_m = ast.Module(body=_body,type_ignores=[]); ast.fix_missing_locations(_m)
ns = {"re":re,"np":np,"json":json,"math":math,"defaultdict":defaultdict}
exec(compile(_m,str(REFERENCE),"exec"), ns, ns)
_missing=[w for w in _want if w not in ns]
if _missing: raise SystemExit(f"FAILED to recover reference helpers: {_missing}")
log("recovered REFERENCE helpers verbatim:", sorted(_want))

def norm(s): return re.sub(r"\s+"," ",str(s or "").lower()).strip()

corpus=[json.loads(l) for l in CORPUS.read_text(encoding="utf-8").splitlines() if l.strip()]
assert len(corpus)==1117
for d in corpus:
    d["lecture_id"]=int(d["lecture_id"]); d["slide_id"]=int(d["slide_id"])
ordered=sorted(corpus,key=lambda d:(d["lecture_id"],d["slide_id"]))
for i,d in enumerate(ordered): d["course_position"]=i
doc_ids=[d["doc_id"] for d in ordered]
pos_of={d["doc_id"]:d["course_position"] for d in ordered}
lec_of={d["doc_id"]:d["lecture_id"] for d in ordered}
text_of={d["doc_id"]:d.get("text","") for d in ordered}

bench=pd.read_json(BENCH,lines=True)
bench=bench[bench.task_type.isin(ANS)].reset_index(drop=True)
tasks=[]
for _,r in bench.iterrows():
    tgt=str(r["target_doc_id"])
    if tgt not in pos_of: continue
    tasks.append({"task_id":str(r["task_id"]),"task_type":r["task_type"],
                  "question":r["question"],"target":tgt,
                  "gold":set(str(x) for x in r["evidence_doc_ids"]),
                  "target_pos":pos_of[tgt],"target_lec":lec_of[tgt]})
tasks=sorted(tasks,key=lambda t:t["task_id"])
assert len(tasks)==600, len(tasks)
sub=[t for i,t in enumerate(tasks) if i % SUBSAMPLE_STRIDE == 0]
log(f"answerable tasks={len(tasks)} prespecified subsample (every {SUBSAMPLE_STRIDE}rd) = {len(sub)}")

# candidate pools: reuse existing saved pool composition is not stored, so
# rebuild pools exactly as the reference advanced script does, from its own ranking code
from sentence_transformers import SentenceTransformer
if torch.cuda.is_available(): raise SystemExit("SAFETY STOP: CUDA visible")
emb_model=SentenceTransformer("BAAI/bge-base-en-v1.5",device="cpu")
doc_emb=emb_model.encode([text_of[d] for d in doc_ids],batch_size=32,
                         convert_to_numpy=True,normalize_embeddings=True,show_progress_bar=True)
log("docs embedded")

# lexical scorers recovered from the advanced script
tok = ns.get("tokenize"); bm25 = ns.get("simple_bm25_scores"); rrf_fn = ns.get("rrf"); doc_text_fn = ns.get("doc_text")
doc_tok=[tok(doc_text_fn(d)) for d in ordered]
from sklearn.feature_extraction.text import TfidfVectorizer
tfv=TfidfVectorizer(lowercase=True,stop_words="english",ngram_range=(1,3),max_df=.85)
tfm=tfv.fit_transform([doc_text_fn(d) for d in ordered])

pools={}
qemb=emb_model.encode([t["question"] for t in sub],batch_size=32,
                      convert_to_numpy=True,normalize_embeddings=True)
for qi,t in enumerate(sub):
    elig=[i for i,d in enumerate(ordered) if d["course_position"]<=t["target_pos"]]
    es=set(elig)
    qt=tok(t["question"])
    bs=bm25(qt,doc_tok)
    tq=tfv.transform([t["question"]]); ts=(tfm@tq.T).toarray().ravel()
    ds=doc_emb@qemb[qi]
    def top(scores,k=50):
        o=np.argsort(-np.asarray(scores),kind="stable")
        return [int(i) for i in o if i in es][:k]
    rb,rt,rg=top(bs),top(ts),top(ds)
    rr=rrf_fn([rb,rt])[:50] if rrf_fn else rb
    rr=[i for i in rr if i in es][:50]
    pools[t["task_id"]]={"bge50":list(dict.fromkeys(rg)),
                         "union50":list(dict.fromkeys(rg+rb+rr))}
log("pools rebuilt")

from transformers import AutoTokenizer, AutoModelForSequenceClassification
rtok=AutoTokenizer.from_pretrained(MODEL)
rmod=AutoModelForSequenceClassification.from_pretrained(MODEL).to("cpu").eval()
if torch.cuda.is_available(): raise SystemExit("SAFETY STOP: CUDA visible after load")
log(f"loaded {MODEL}")

def chunks(txt):
    w=str(txt or "").split()
    if len(w)<=CHUNK_WORDS: return [" ".join(w)]
    return [" ".join(w[i:i+CHUNK_WORDS]) for i in range(0,len(w),CHUNK_STRIDE)] or [" ".join(w)]

def score_pairs(pairs):
    out=np.zeros(len(pairs),dtype=float); t0=time.time()
    nb=math.ceil(len(pairs)/BATCH)
    with torch.inference_mode():
        for b,s0 in enumerate(range(0,len(pairs),BATCH),1):
            bp=pairs[s0:s0+BATCH]
            tk=rtok([q for q,_ in bp],[d for _,d in bp],padding=True,
                    truncation=True,max_length=MAXLEN,return_tensors="pt")
            out[s0:s0+len(bp)]=rmod(**tk).logits.view(-1).float().cpu().numpy()
            if b==1 or b%200==0 or b==nb:
                el=time.time()-t0; rate=(s0+len(bp))/max(el,1e-9)
                log(f"  batch {b}/{nb} pairs {s0+len(bp)}/{len(pairs)} "
                    f"elapsed={el/60:.1f}m ETA={(len(pairs)-s0-len(bp))/max(rate,1e-9)/60:.1f}m")
    return out

results={}
for policy in ["truncate512"]:  # analysis stage SCOPE: paper-cited variant only; chunked_max full-600 not run (see 09 report)
    for pname in ["bge50","union50"]:
        pairs=[]; index=[]
        for t in sub:
            for di in pools[t["task_id"]][pname]:
                did=doc_ids[di]
                if policy=="truncate512":
                    pairs.append((t["question"],text_of[did])); index.append((t["task_id"],did,0))
                else:
                    for ci,ch in enumerate(chunks(text_of[did])):
                        pairs.append((t["question"],ch)); index.append((t["task_id"],did,ci))
        log(f"\n[{policy}/{pname}] scoring {len(pairs)} pairs")
        sc=score_pairs(pairs)
        agg={}
        for (tid,did,_),v in zip(index,sc):
            k=(tid,did); agg[k]=max(agg.get(k,-1e9),float(v))
        rk={}
        for t in sub:
            cand=[doc_ids[i] for i in pools[t["task_id"]][pname]]
            rk[t["task_id"]]=sorted(cand,key=lambda d:(-agg[(t["task_id"],d)],d))
        results[f"{policy}__{pname}"]=rk
        log(f"[{policy}/{pname}] done")

def evaluate(rank_map,label):
    rows=[];mr=[]
    for t in sub:
        r=rank_map[t["task_id"]]
        fg=next((i+1 for i,d in enumerate(r) if d in t["gold"]),None)
        ft=next((i+1 for i,d in enumerate(r) if d==t["target"]),None)
        mr.append({"task_id":t["task_id"],"task_type":t["task_type"],"method":label,
                   "first_gold_rank":fg,"first_target_rank":ft,
                   "rr_any_gold":1/fg if fg else 0.0,"rr_target":1/ft if ft else 0.0})
        for k in KS:
            top=set(r[:k])
            rows.append({"method":label,"k":k,"task_id":t["task_id"],
                "any_gold":int(bool(top&t["gold"])),
                "all_gold":int(bool(t["gold"]) and t["gold"]<=top),
                "target_slide":int(t["target"] in top),
                "target_lecture":int(any(lec_of[d]==t["target_lec"] for d in r[:k]))})
    return pd.DataFrame(rows),pd.DataFrame(mr)

ks_all=[];mr_all=[]
for lab,rk in results.items():
    a,b=evaluate(rk,f"v2m3_{lab}"); ks_all.append(a); mr_all.append(b)

# matched comparison: base reranker + RRF/BM25 on the SAME 200 tasks
prev_ks=pd.read_csv(ROOT/"artifacts/retrieval_diagnostics/advanced_retrieval_k_sweep_detail.csv")
subids=set(t["task_id"] for t in sub)
prev=prev_ks[prev_ks.task_id.isin(subids)&prev_ks.method.isin(
    ["bm25","rrf","bge_dense","bge_reranker_bge50","bge_reranker_union50"])]
ks_all.append(prev[["method","k","task_id","any_gold","all_gold","target_slide","target_lecture"]])
prev_mr=pd.read_csv(ROOT/"artifacts/retrieval_diagnostics/advanced_retrieval_mrr_detail.csv")
prev_mr=prev_mr[prev_mr.task_id.isin(subids)&prev_mr.method.isin(
    ["bm25","rrf","bge_dense","bge_reranker_bge50","bge_reranker_union50"])]
mr_all.append(prev_mr[["task_id","task_type","method","first_gold_rank","first_target_rank","rr_any_gold","rr_target"]])

K=pd.concat(ks_all,ignore_index=True); M=pd.concat(mr_all,ignore_index=True)
sweep=(K.groupby(["method","k"],as_index=False)
       .agg(N=("task_id","nunique"),AnyGoldRecall=("any_gold","mean"),
            AllGoldRecall=("all_gold","mean"),TargetSlideRecall=("target_slide","mean"),
            TargetLectureRecall=("target_lecture","mean")))
sweep.to_csv(OUT/"full_v2m3_k_sweep.csv",index=False)
mrr=(M.groupby("method",as_index=False)
     .agg(N=("task_id","nunique"),MRRAnyGold=("rr_any_gold","mean"),
          MRRTargetSlide=("rr_target","mean"),
          MedianFirstGoldRank=("first_gold_rank","median"),
          MedianFirstTargetRank=("first_target_rank","median")))
mrr.to_csv(OUT/"full_v2m3_mrr.csv",index=False)
import platform, subprocess
_rev=lambda m: getattr(getattr(rmod,"config",None),"_name_or_path",m)
json.dump({"model":MODEL,"baseline_model":BASE_MODEL,
           "tokenizer_class":type(rtok).__name__,"model_config_name":_rev(MODEL),
           "hardware":"CPU only (both host GPUs occupied by another user's vLLM workers; not used)",
           "platform":platform.platform(),"full_600_tasks":True,"subsample_stride":SUBSAMPLE_STRIDE,
           "n_tasks":len(sub),"batch":BATCH,"max_length":MAXLEN,
           "chunk_words":CHUNK_WORDS,"chunk_stride":CHUNK_STRIDE,
           "generation_performed":False,"cpu_only":True},
          open(OUT/"full_v2m3_run_metadata.json","w"),indent=2)
log("\n"+sweep[sweep.k.isin([3,10,50])].to_string(index=False))
log("\n"+mrr.to_string(index=False))
log("STRONGER_RERANKER = DONE (retrieval only)")
