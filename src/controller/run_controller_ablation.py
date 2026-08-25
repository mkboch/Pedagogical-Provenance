#!/usr/bin/env python3
"""PHASES 5 + 6.

5. Apply the FROZEN controller pipeline to the 600 real answerable
   Benchmark-v1 QA questions -> answerable-QA false intervention rate.
6. Concept-held-out feature ablation on Canonical Multi-Concept Sequence Set, reusing the EXACT
   existing fold assignments, plus a non-learned lexical lookup rule.

No task labels are used as input. No LLM generation. CPU only.
"""
from pathlib import Path
import ast, json, re
import numpy as np, pandas as pd, torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (confusion_matrix, precision_recall_fscore_support,
                             roc_auc_score, balanced_accuracy_score, accuracy_score, f1_score)

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/"artifacts/final_targeted_validation_20260822"; OUT.mkdir(parents=True, exist_ok=True)
PREV = ROOT/"artifacts/sequence_policy"
SRC = ROOT/"src/controller/run_course_aware_controller.py"
CORPUS = ROOT/"data/corpus/slide_corpus_final.jsonl"
BENCH = ROOT/"data/benchmark/core_qa_benchmark.jsonl"
SEED = 20260817
EMBED = "BAAI/bge-base-en-v1.5"
QP = "Represent this sentence for searching relevant passages: "
ANS = {"evidence_cited_qa","slide_local_factual_qa","neighbor_slide_conceptual_qa"}

FEATURES = ["lex_log_past_docs","lex_log_future_docs","lex_future_fraction",
 "sem_past_max","sem_future_max","sem_delta_future_past","sem_past_top3",
 "sem_future_top3","sem_delta_top3","sem_top10_future_fraction",
 "sem_first_available_rank","position_fraction"]
LEXF = ["lex_log_past_docs","lex_log_future_docs","lex_future_fraction"]
SEMF = ["sem_past_max","sem_future_max","sem_delta_future_past","sem_past_top3",
        "sem_future_top3","sem_delta_top3","sem_top10_future_fraction","sem_first_available_rank"]
POSF = ["position_fraction"]

def log(*a): print(*a, flush=True)

# recover ALL top-level controller functions verbatim (transitive closure)
tree = ast.parse(SRC.read_text(encoding="utf-8"))
keep=[]
for n in tree.body:
    if isinstance(n,(ast.Assign,ast.AnnAssign)):
        try: ast.literal_eval(n.value); keep.append(n)
        except Exception: pass
keep += [n for n in tree.body if isinstance(n, ast.FunctionDef)]
mod = ast.Module(body=keep, type_ignores=[]); ast.fix_missing_locations(mod)
ns = {"re":re,"np":np,"json":json,"math":__import__("math")}
exec(compile(mod,str(SRC),"exec"), ns, ns)
strip_cues, extract_concept = ns["strip_cues"], ns["extract_concept"]
clinical_intent, phrase_pattern, top_mean = ns["clinical_intent"], ns["phrase_pattern"], ns["top_mean"]
log("recovered controller helpers verbatim")

def norm(s): return re.sub(r"\s+"," ",str(s or "").lower()).strip()
corpus=[json.loads(l) for l in CORPUS.read_text(encoding="utf-8").splitlines() if l.strip()]
for d in corpus:
    d["lecture_id"]=int(d["lecture_id"]); d["slide_id"]=int(d["slide_id"]); d["_n"]=norm(d.get("text",""))
ordered=sorted(corpus,key=lambda d:(d["lecture_id"],d["slide_id"]))
for i,d in enumerate(ordered): d["cp"]=i
pos_of={d["doc_id"]:d["cp"] for d in ordered}
maxlec=max(d["lecture_id"] for d in ordered)

from sentence_transformers import SentenceTransformer
if torch.cuda.is_available(): log("NOTE: forcing CPU (matches frozen controller protocol)")
model=SentenceTransformer(EMBED, device="cpu")
doc_emb=model.encode([d.get("text","") for d in ordered],batch_size=32,
                     convert_to_numpy=True,normalize_embeddings=True,show_progress_bar=True)
log("corpus embedded")

def featurize(question, anchor_pos, lecture):
    stripped=strip_cues(question)
    concept=extract_concept(stripped)
    pat=phrase_pattern(concept) if concept else None
    lp=lf=0
    if pat is not None:
        for d in ordered:
            if pat.search(d["_n"]):
                if d["cp"]<=anchor_pos: lp+=1
                else: lf+=1
    tot=lp+lf
    ce=model.encode([QP+(concept or question)],convert_to_numpy=True,normalize_embeddings=True)[0]
    sims=doc_emb@ce
    pm=np.array([d["cp"]<=anchor_pos for d in ordered])
    ps,fs=sims[pm],sims[~pm]; order=np.argsort(-sims)
    fa=next((r+1 for r,i in enumerate(order) if pm[i]), len(sims))
    return {"concept_extracted":concept,"clinical_rule":int(clinical_intent(stripped)),
      "lex_past_docs":lp,"lex_future_docs":lf,
      "lex_log_past_docs":float(np.log1p(lp)),"lex_log_future_docs":float(np.log1p(lf)),
      "lex_future_fraction":float(lf/tot) if tot else 0.0,
      "sem_past_max":float(ps.max()) if ps.size else 0.0,
      "sem_future_max":float(fs.max()) if fs.size else 0.0,
      "sem_delta_future_past":float((fs.max() if fs.size else 0)-(ps.max() if ps.size else 0)),
      "sem_past_top3":float(top_mean(ps)) if ps.size else 0.0,
      "sem_future_top3":float(top_mean(fs)) if fs.size else 0.0,
      "sem_delta_top3":float((top_mean(fs) if fs.size else 0)-(top_mean(ps) if ps.size else 0)),
      "sem_top10_future_fraction":float(np.mean([not pm[i] for i in order[:10]])),
      "sem_first_available_rank":float(fa),
      "position_fraction":float(lecture/max(maxlec,1))}

# =====================================================================
# real answerable QA
# =====================================================================
bench=pd.read_json(BENCH,lines=True)
aq=bench[bench.task_type.isin(ANS)].copy()
log(f"\nFeaturizing {len(aq)} answerable QA questions")
rows=[]
for i,(_,r) in enumerate(aq.iterrows(),1):
    tgt=str(r.target_doc_id); ap=pos_of[tgt]
    lec=int([d for d in ordered if d["doc_id"]==tgt][0]["lecture_id"])
    f=featurize(r.question, ap, lec)
    f.update({"task_id":str(r.task_id),"task_type":r.task_type,
              "anchor_course_position":ap,"anchor_lecture":lec})
    rows.append(f)
    if i%150==0: log(f"  {i}/{len(aq)}")
real=pd.DataFrame(rows)

# train the sequence model on Canonical Multi-Concept Sequence Set (its native training population),
# then score real answerable QA. Clinical branch applied exactly as deployed.
ext=pd.read_csv(PREV/"concept_grouped_controller_features.csv")
pipe=Pipeline([("sc",StandardScaler()),("lr",LogisticRegression(C=1.0,class_weight="balanced",
      max_iter=3000,random_state=SEED))])
pipe.fit(ext[FEATURES].to_numpy(float),(ext.sequence_label=="not_yet_available").astype(int).to_numpy())
real["seq_prob"]=pipe.predict_proba(real[FEATURES].to_numpy(float))[:,1]
p5=[]
for thr in [0.50,0.60,0.65,0.70,0.75]:
    seq=(real.seq_prob>=thr).astype(int)
    interv=((real.clinical_rule==1)|(seq==1)).astype(int)
    p5.append({"threshold":thr,"N":len(real),
       "false_intervention_rate":float(interv.mean()),
       "n_false_intervention":int(interv.sum()),
       "via_clinical_rule":int((real.clinical_rule==1).sum()),
       "via_sequence_branch":int(((real.clinical_rule==0)&(seq==1)).sum()),
       "answer_preservation":float(1-interv.mean())})
    for t,g in real.assign(i=interv).groupby("task_type"):
        p5[-1][f"fi_{t}"]=float(g.i.mean())
p5=pd.DataFrame(p5)
real.to_csv(OUT/"controller_real_answerable_predictions.csv",index=False)
p5.to_csv(OUT/"controller_real_answerable_by_threshold.csv",index=False)
log("\nController evaluation on 600 answerable QA tasks:")
log(p5.to_string(index=False))

# =====================================================================
# feature ablation with EXACT existing folds
# =====================================================================
log("\nFeature ablation (concept-held-out, same folds)")
X_all=ext[FEATURES].to_numpy(float); y=(ext.sequence_label=="not_yet_available").astype(int).to_numpy()
groups=ext.concept_canonical.to_numpy()
sgkf=StratifiedGroupKFold(n_splits=5,shuffle=True,random_state=SEED)
folds=list(sgkf.split(X_all,y,groups))

SETS={"1_all_features":FEATURES,"2_lexical_only":LEXF,"3_semantic_only":SEMF,
 "4_position_only":POSF,"5_lexical_plus_position":LEXF+POSF,
 "6_semantic_plus_position":SEMF+POSF,
 "7_all_except_exact_match_counts":[f for f in FEATURES if f not in LEXF]}

def evaluate(name,yv,pred,prob=None,foldrows=None):
    cm=confusion_matrix(yv,pred,labels=[0,1]); tn,fp,fn,tp=cm.ravel()
    pr,rc,f1,sup=precision_recall_fscore_support(yv,pred,labels=[0,1],zero_division=0)
    d={"system":name,"accuracy":accuracy_score(yv,pred),
     "balanced_accuracy":balanced_accuracy_score(yv,pred),"macro_f1":f1_score(yv,pred,average="macro"),
     "available_precision":pr[0],"available_recall":rc[0],"available_f1":f1[0],
     "notyet_precision":pr[1],"notyet_recall":rc[1],"notyet_f1":f1[1],
     "sequence_recall":rc[1],"false_positive_rate_on_available":fp/(fp+tn) if (fp+tn) else np.nan,
     "TP":int(tp),"FP":int(fp),"FN":int(fn),"TN":int(tn)}
    d["roc_auc"]=roc_auc_score(yv,prob) if prob is not None else np.nan
    return d

abl=[]; foldrows=[]
for name,feats in SETS.items():
    X=ext[feats].to_numpy(float); oof=np.full(len(y),np.nan)
    for k,(tr,te) in enumerate(folds,1):
        pl=Pipeline([("sc",StandardScaler()),("lr",LogisticRegression(C=1.0,class_weight="balanced",
              max_iter=3000,random_state=SEED))])
        pl.fit(X[tr],y[tr]); oof[te]=pl.predict_proba(X[te])[:,1]
        pk=(oof[te]>=0.5).astype(int)
        foldrows.append({"system":name,"fold":k,"n_test":len(te),
            "accuracy":accuracy_score(y[te],pk),"balanced_accuracy":balanced_accuracy_score(y[te],pk),
            "macro_f1":f1_score(y[te],pk,average="macro")})
    abl.append(evaluate(name,y,(oof>=0.5).astype(int),oof))

# NON-LEARNED lookup rule: available iff prior-course lexical support > 0
lookup_pred=(ext.lex_past_docs.to_numpy()==0).astype(int)   # 1 == predict "not_yet_available"
abl.append(evaluate("8_NONLEARNED_lookup_lex_past_docs_gt_0",y,lookup_pred,None))
for k,(tr,te) in enumerate(folds,1):
    foldrows.append({"system":"8_NONLEARNED_lookup_lex_past_docs_gt_0","fold":k,"n_test":len(te),
        "accuracy":accuracy_score(y[te],lookup_pred[te]),
        "balanced_accuracy":balanced_accuracy_score(y[te],lookup_pred[te]),
        "macro_f1":f1_score(y[te],lookup_pred[te],average="macro")})

A=pd.DataFrame(abl); A.to_csv(OUT/"controller_feature_ablation.csv",index=False)
pd.DataFrame(foldrows).to_csv(OUT/"controller_feature_ablation_fold_results.csv",index=False)
pd.set_option("display.width",250)
log("\nFeature ablation results (concept-held-out, identical folds):")
log(A[["system","accuracy","balanced_accuracy","macro_f1","sequence_recall",
       "false_positive_rate_on_available","roc_auc"]].round(4).to_string(index=False))
log("\nCONTROLLER_REALQA_AND_ABLATION = DONE")
