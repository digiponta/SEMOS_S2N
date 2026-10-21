"""SEMOS_S2N v0.9.2: conservative proposition-aware retrieval gate.

Rule baseline for a narrow Japanese GPU/CPU example domain, not an NLI model.
Never conclude KNOWN from cosine alone when parsing is uncertain.
"""
from __future__ import annotations
import argparse
import json
import re
from pathlib import Path
from semos_benchmark_v091 import evaluate as baseline_evaluate
from semos_real_adapter_v07 import encode_japanese
from semos_memory_v09 import cosine

def proposition(text):
    s=re.sub(r"\s+","",text).rstrip("。")
    subject=re.search(r"(?<![A-Za-z])(GPU|CPU)(?![A-Za-z])",s)
    if not subject:
        return None
    subj=subject.group(1)
    # Recognize only explicitly specified constructions. Negation is scoped
    # to the relevant predicate, not guessed from any occurrence of 'ない'.
    if ("並列計算" in s or "並行演算" in s):
        relation="parallel-computing"
        if re.search(r"適していない|向いていない|実行できない",s): polarity=False
        elif re.search(r"適している|向いている|効率よく実行できる",s): polarity=True
        else: return None
    elif "機械学習" in s:
        relation="machine-learning-utility"
        if re.search(r"有用でない|役立たない|有用ではない",s): polarity=False
        elif re.search(r"有用である|役立つ",s): polarity=True
        else: return None
    else:
        return None
    return {"subject":subj,"relation":relation,"polarity":polarity}

def gate(reference, query, similarity, threshold=0.85):
    ref=proposition(reference)
    asked=proposition(query)
    if ref is None or asked is None:
        return {"decision":"REVIEW","reason":"unsupported-or-ambiguous-proposition"}
    if ref["subject"] != asked["subject"]:
        return {"decision":"UNKNOWN","reason":"subject-mismatch"}
    if ref["relation"] != asked["relation"]:
        return {"decision":"UNKNOWN","reason":"relation-mismatch"}
    if ref["polarity"] != asked["polarity"]:
        return {"decision":"UNKNOWN","reason":"polarity-mismatch"}
    if similarity >= threshold:
        return {"decision":"KNOWN","reason":"structure-and-vector-agree"}
    return {"decision":"REVIEW","reason":"structure-match-low-similarity"}

def evaluate(cases, models, *, threshold=0.85, device="auto", encoder=encode_japanese):
    if not isinstance(cases,list) or not cases: raise ValueError("cases required")
    vectors={}
    output=[]
    for row in cases:
        receiver=row["receiver"];spec=models[receiver]
        key=(receiver,row["reference"])
        if key not in vectors:
            vectors[key]=encoder(row["reference"],checkpoint=spec["checkpoint"],tokenizer_path=spec["tokenizer"],device=device)
        query=encoder(row["query"],checkpoint=spec["checkpoint"],tokenizer_path=spec["tokenizer"],device=device)
        reference=vectors[key]
        if (reference["encoder_sha256"],reference["tokenizer_sha256"]) != (query["encoder_sha256"],query["tokenizer_sha256"]):
            raise ValueError("mixed model fingerprint within one receiver")
        similarity=cosine(reference["vector"],query["vector"])
        prediction=gate(row["reference"],row["query"],similarity,threshold)
        output.append({**row,"cosine":round(similarity,6),**prediction,
                       "correct":prediction["decision"]==row["expected"]})
    known=[r for r in output if r["expected"]=="KNOWN"]
    unknown=[r for r in output if r["expected"]=="UNKNOWN"]
    recall=sum(r["correct"] for r in known)/len(known) if known else None
    reject=sum(r["correct"] for r in unknown)/len(unknown) if unknown else None
    review=sum(r["decision"]=="REVIEW" for r in output)
    return {"version":"v0.9.2","gate":"restricted-proposition-rules+receiver-local-cosine",
            "threshold":threshold,"sample_count":len(output),
            "known_recall":recall,"unknown_reject_rate":reject,
            "balanced_accuracy":(recall+reject)/2 if recall is not None and reject is not None else None,
            "review_count":review,"coverage":1-review/len(output),
            "evaluation_status":"illustrative-in-sample-rules-not-independent-holdout",
            "results":output}

def main():
    p=argparse.ArgumentParser(description="Restricted proposition truth-aware gate evaluation")
    p.add_argument("cases");p.add_argument("models");p.add_argument("output")
    p.add_argument("--device",choices=["auto","cpu","cuda"],default="auto")
    p.add_argument("--threshold",type=float,default=0.85)
    a=p.parse_args()
    try:
        cases=json.loads(Path(a.cases).read_text(encoding="utf-8-sig"))
        models=json.loads(Path(a.models).read_text(encoding="utf-8-sig"))
        result=evaluate(cases,models,device=a.device,threshold=a.threshold)
        out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True)
        out.write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
        print(json.dumps(result,ensure_ascii=False,indent=2))
    except (OSError,ValueError,TypeError,KeyError) as exc:
        p.exit(2,f"error: {exc}\n")
if __name__=="__main__":main()
