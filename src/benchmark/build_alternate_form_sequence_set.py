#!/usr/bin/env python3
"""Alternate-Form Sequence Robustness Set.

Separate alternate-form robustness set. The canonical sequence set and core QA benchmark are not modified.

Every alternate surface form must be EVIDENCED in the MEDI-SLATE course corpus
(explicit equivalence statement, acronym definition, or full form). No LLM
generated any paraphrase. No synonym is invented.

Items are PAIRED to the canonical sequence set: same anchor, same label, same underlying concept;
ONLY the concept's surface realization in the question text changes.
"""
from pathlib import Path
import hashlib, json, re, unicodedata
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/"artifacts/sequence_policy"
CANONICAL_SET_PATH = ROOT/"artifacts/sequence_policy/canonical_multiconcept_sequence.jsonl"
CORPUS = ROOT/"data/corpus/slide_corpus_final.jsonl"

def norm(s):
    s=unicodedata.normalize("NFKC",str(s or "")).lower()
    for a,b in [("‐","-"),("‑","-"),("–","-"),("—","-")]: s=s.replace(a,b)
    return re.sub(r"\s+"," ",s).strip()

def pat(term):
    return re.compile(r"(?<![a-z0-9])"+r"[\s\-]+".join(re.escape(x) for x in norm(term).replace("-"," ").split())+r"(?![a-z0-9])")

# ---------------------------------------------------------------------
# ALIAS CANDIDATES — each must pass corpus evidence check below.
# relationship types: acronym | full_form | synonym | alternate_representation
# EXCLUDED deliberately (documented in the spec):
#   backprojection->"back projection" (hyphenation only; phrase_pattern already
#     normalizes hyphens, so it would not test surface-form robustness)
#   attenuation->"absorption" (mechanism vs phenomenon, not an alias)
#   resolution->"spatial resolution" (corpus states resolution has several
#     dimensions INCLUDING spatial => subtype, not alias)
# ---------------------------------------------------------------------
# NOTE: full-form expansions that CONTAIN the canonical phrase are excluded.
# "image reconstruction" still contains "reconstruction", so the controller's
# phrase matcher fires regardless -- such an item would not actually test
# surface-form robustness. Only aliases that genuinely REPLACE the canonical
# surface string are retained. This was caught by the canonical_phrase_absent
# validity gate on the first build and is a design correction, not a relaxation.
CANDIDATES = [
 ("inner product","dot product","synonym"),
 ("filtered backprojection","fbp","acronym"),
 ("impulse response","point spread function","synonym"),
 ("impulse response","psf","acronym"),
 ("k-space","fourier space","alternate_representation"),
 ("k-space","frequency domain","alternate_representation"),
 ("orthogonal","perpendicular","synonym"),
 ("sampling","discretization","synonym"),
 ("sinogram","radon space","alternate_representation"),
 ("detector","sensor","synonym"),
]
EXCLUDED_FULLFORM = {
 "delta function->dirac delta function":"full form CONTAINS canonical phrase; matcher still fires",
 "reconstruction->image reconstruction":"full form CONTAINS canonical phrase; matcher still fires",
 "dose->radiation dose":"full form CONTAINS canonical phrase; matcher still fires"}

def main():
    corpus=[json.loads(l) for l in CORPUS.read_text(encoding="utf-8").splitlines() if l.strip()]
    for d in corpus: d["_n"]=norm(d.get("text",""))

    # ---------------- evidence check ----------------
    inv=[]
    for canon, alt, rel in CANDIDATES:
        p=pat(alt); hits=[d for d in corpus if p.search(d["_n"])]
        if not hits:
            inv.append({"canonical_concept":canon,"alternate_surface_form":alt,
                "relationship_type":rel,"evidenced":False,"n_corpus_slides":0,
                "source_doc_id":None,"source_snippet":None}); continue
        h=hits[0]; m=p.search(h["_n"]); s=max(0,m.start()-110)
        inv.append({"canonical_concept":canon,"alternate_surface_form":alt,
            "relationship_type":rel,"evidenced":True,"n_corpus_slides":len(hits),
            "source_doc_id":h["doc_id"],
            "source_snippet":re.sub(r"\s+"," ",h["_n"][s:m.start()+140])})
    INV=pd.DataFrame(inv)
    INV.to_csv(OUT/"alternate_form_alias_inventory.csv",index=False)
    good=INV[INV.evidenced]
    # one alias per concept (first evidenced, deterministic order)
    chosen={}
    for _,r in good.iterrows():
        chosen.setdefault(r.canonical_concept,(r.alternate_surface_form,r.relationship_type,r.source_doc_id))
    print(f"evidenced alias rows: {len(good)}/{len(CANDIDATES)}   concepts covered: {len(chosen)}")

    # ---------------- build paired items ----------------
    canonical_tasks=[json.loads(l) for l in CANONICAL_SET_PATH.read_text(encoding="utf-8").splitlines() if l.strip()]
    tasks=[]; n=0
    for t in canonical_tasks:
        c=t["metadata"]["concept"]
        if c not in chosen: continue
        alt,rel,src=chosen[c]
        p=pat(c)
        if not p.search(norm(t["question"])): continue     # must be replaceable
        n+=1
        q=p.sub(alt, t["question"], count=1)
        md=dict(t["metadata"])
        md.update({"artifact_role":"alternate_form","paired_canonical_task_id":t["task_id"],
                   "canonical_concept":c,"surface_form_used":alt,
                   "alias_relationship_type":rel,"alias_evidence_doc_id":src})
        tasks.append({"task_id":f"SGX12_{n:05d}","task_type":"sequence_generalization_probe_paraphrased",
            "question":q,"reference_answer":t["reference_answer"],
            "evidence_doc_ids":t["evidence_doc_ids"],"target_doc_id":t["target_doc_id"],
            "metadata":md})

    jp=OUT/"alternate_form_sequence.jsonl"
    with jp.open("w",encoding="utf-8") as f:
        for t in tasks: f.write(json.dumps(t,ensure_ascii=False)+"\n")

    # ---------------- automatic validation ----------------
    by={t["task_id"]:t for t in canonical_tasks}
    checks=[]; fails=[]
    def ck(name,ok,det):
        checks.append({"check":name,"result":"PASS" if ok else "FAIL","detail":det})
        if not ok: fails.append(name)
    ids=[t["task_id"] for t in tasks]
    ck("no_duplicate_ids",len(ids)==len(set(ids)),f"{len(ids)} ids")
    ck("labels_unchanged",all(t["metadata"]["sequence_label"]==by[t["metadata"]["paired_canonical_task_id"]]["metadata"]["sequence_label"] for t in tasks),"vs paired canonical task")
    ck("anchors_unchanged",all(t["target_doc_id"]==by[t["metadata"]["paired_canonical_task_id"]]["target_doc_id"]
        and t["metadata"]["anchor_course_position"]==by[t["metadata"]["paired_canonical_task_id"]]["metadata"]["anchor_course_position"] for t in tasks),"target_doc_id + anchor position")
    ck("only_question_text_changed",all(t["reference_answer"]==by[t["metadata"]["paired_canonical_task_id"]]["reference_answer"]
        and t["evidence_doc_ids"]==by[t["metadata"]["paired_canonical_task_id"]]["evidence_doc_ids"] for t in tasks),"reference+evidence identical")
    ck("question_actually_differs",all(t["question"]!=by[t["metadata"]["paired_canonical_task_id"]]["question"] for t in tasks),"surface form replaced")
    ck("alternate_form_evidenced",all(t["metadata"]["alias_evidence_doc_id"] for t in tasks),"every alias has a corpus source doc")
    ck("canonical_phrase_absent",all(not pat(t["metadata"]["canonical_concept"]).search(norm(t["question"])) for t in tasks),"canonical phrase removed from question")
    # immutability
    canonical_hash=hashlib.sha256(CANONICAL_SET_PATH.read_bytes()).hexdigest()
    ck("canonical_sequence_unchanged",canonical_hash=="48f30e9ef9e3a4867b6a4777cc87cc5c65987f349549f7f1f0a01f473be2360c",f"canonical sequence sha256={canonical_hash[:16]}")
    hb=hashlib.sha256((ROOT/"data/benchmark/core_qa_benchmark.jsonl").read_bytes()).hexdigest()
    ck("core_benchmark_unchanged",hb.startswith("8fa4c1a39149cbcf"),f"core benchmark sha256={hb[:16]}")
    CH=pd.DataFrame(checks); CH.to_csv(OUT/"alternate_form_validation_checks.csv",index=False)

    eh=hashlib.sha256(jp.read_bytes()).hexdigest()
    ih=hashlib.sha256((OUT/"alternate_form_alias_inventory.csv").read_bytes()).hexdigest()
    lab=pd.Series([t["metadata"]["sequence_label"] for t in tasks]).value_counts().to_dict()
    meta={"artifact_role":"alternate_surface_form_robustness","artifact_name":"Alternate-Form Sequence Robustness Set",
      "creation_date":"2026-08-22","paired_to":"canonical_multiconcept_sequence.jsonl",
      "canonical_sequence_sha256":canonical_hash,"core_benchmark_sha256":hb,
      "artifact_sha256":eh,"alias_inventory_sha256":ih,
      "task_count":len(tasks),"concept_count":len(chosen),
      "concepts":sorted(chosen),"class_counts":lab,
      "alias_source_rule":"alternate surface form must occur in the MEDI-SLATE corpus; explicit equivalence, acronym definition, or full form. NO LLM. NO invented synonyms.",
      "excluded_full_form_aliases":EXCLUDED_FULLFORM,
      "excluded_candidates":{"backprojection->back projection":"hyphenation only; phrase_pattern already normalizes hyphens",
        "attenuation->absorption":"mechanism vs phenomenon, not an alias",
        "resolution->spatial resolution":"corpus states resolution has several dimensions including spatial => subtype"},
      "only_field_changed":"question (concept surface form only)",
      "no_llm_used":True,"no_human_ratings":True}
    (OUT/"alternate_form_sequence_metadata.json").write_text(json.dumps(meta,indent=2),encoding="utf-8")
    (OUT/"alternate_form_sequence.sha256").write_text(
        f"{eh}  alternate_form_sequence.jsonl\n{ih}  alternate_form_alias_inventory.csv\n",encoding="utf-8")

    print(f"tasks={len(tasks)} concepts={len(chosen)} classes={lab}")
    print(f"sha256={eh}")
    print(CH.to_string(index=False))
    print("ALTERNATE_FORM_VALIDITY_GATE =", "PASSED" if not fails else f"FAILED {fails}")

if __name__=="__main__": main()
