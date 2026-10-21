"""SEMOS_S2N v0.9.1: labeled paraphrase, contradiction and unknown benchmark.

Uses receiver-local LLM_SEM embedding retrieval; does not treat cosine as
proof of entailment. Threshold is fixed before evaluation.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from semos_memory_v09 import LocalMemory
from semos_real_adapter_v07 import encode_japanese

CATEGORIES={"exact","paraphrase","contradiction","subject_change","unrelated"}
def evaluate(cases, models, *, threshold=0.85, device="auto", encoder=encode_japanese):
    if not isinstance(cases,list) or not cases: raise ValueError("nonempty cases required")
    memories={}
    results=[]
    for row in cases:
        receiver=row["receiver"]
        if receiver not in models: raise ValueError("missing receiver model: "+receiver)
        if row.get("category") not in CATEGORIES: raise ValueError("invalid category")
        if row.get("expected") not in ("KNOWN","UNKNOWN"): raise ValueError("expected label required")
        spec=models[receiver]
        if receiver not in memories:
            memories[receiver]=LocalMemory(receiver,threshold)
        memory=memories[receiver]
        reference=row["reference"]
        if not memory.records:
            base=encoder(reference,checkpoint=spec["checkpoint"],tokenizer_path=spec["tokenizer"],device=device)
            memory.insert(reference,base)
        elif not any(r["text"]==reference for r in memory.records):
            raise ValueError("each receiver must have one fixed reference in this benchmark")
        vector=encoder(row["query"],checkpoint=spec["checkpoint"],tokenizer_path=spec["tokenizer"],device=device)
        prediction=memory.query(row["query"],vector)
        results.append({"receiver":receiver,"category":row["category"],"reference":reference,
                        "query":row["query"],"expected":row["expected"],"decision":prediction["decision"],
                        "score":prediction["score"],"correct":prediction["decision"]==row["expected"]})
    known=[r for r in results if r["expected"]=="KNOWN"]
    unknown=[r for r in results if r["expected"]=="UNKNOWN"]
    per_category={}
    for category in sorted(CATEGORIES):
        subset=[r for r in results if r["category"]==category]
        if subset:
            per_category[category]={"count":len(subset),"accuracy":sum(x["correct"] for x in subset)/len(subset)}
    known_recall=sum(x["correct"] for x in known)/len(known) if known else None
    unknown_reject=sum(x["correct"] for x in unknown)/len(unknown) if unknown else None
    balanced=(known_recall+unknown_reject)/2 if known and unknown else None
    return {"version":"v0.9.1","threshold":threshold,"sample_count":len(results),
            "known_recall":known_recall,"unknown_reject_rate":unknown_reject,
            "balanced_accuracy":balanced,"per_category":per_category,
            "semantic_equivalence":"not-independently-verified",
            "notes":"Fixed benchmark with human-assigned labels; cosine baseline is not a truth gate",
            "results":results}

def main():
    p=argparse.ArgumentParser(description="Labeled semantic retrieval benchmark")
    p.add_argument("cases");p.add_argument("models");p.add_argument("output")
    p.add_argument("--threshold",type=float,default=0.85)
    p.add_argument("--device",choices=["cpu","cuda","auto"],default="auto")
    a=p.parse_args()
    try:
        cases=json.loads(Path(a.cases).read_text(encoding="utf-8-sig"))
        models=json.loads(Path(a.models).read_text(encoding="utf-8-sig"))
        report=evaluate(cases,models,threshold=a.threshold,device=a.device)
        out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True)
        out.write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
        print(json.dumps(report,ensure_ascii=False,indent=2))
    except (ValueError,TypeError,KeyError,OSError) as exc:
        p.exit(2,f"error: {exc}\n")
if __name__=="__main__":main()
