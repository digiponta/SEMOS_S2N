"""SEMOS_S2N v0.9.3 frozen holdout evaluator.

Run v0.9.2 unmodified. Report selective accuracy and abstentions separately.
The authored holdout is new to the v0.9.2 rule design, not a random sample
of Japanese language or evidence of statistical generalization.
"""
import argparse
import json
from collections import Counter
from pathlib import Path
from semos_truth_gate_v092 import evaluate as evaluate_gate

def summarize(report):
    rows=report["results"]
    accepted=[r for r in rows if r["decision"]!="REVIEW"]
    labeled=[r for r in rows if r["expected"] in ("KNOWN","UNKNOWN")]
    if len(labeled)!=len(rows): raise ValueError("all holdout examples must be labeled")
    known=[r for r in rows if r["expected"]=="KNOWN"]
    unknown=[r for r in rows if r["expected"]=="UNKNOWN"]
    false_known=[r for r in unknown if r["decision"]=="KNOWN"]
    def fraction(n,d): return n/d if d else None
    by_category={}
    for cat in sorted(set(r["category"] for r in rows)):
        group=[r for r in rows if r["category"]==cat]
        by_category[cat]={"count":len(group),
            "coverage":fraction(sum(r["decision"]!="REVIEW" for r in group),len(group)),
            "strict_accuracy":fraction(sum(r["correct"] for r in group),len(group))}
    return {
        "version":"v0.9.3",
        "data_status":"separate-authored-development-holdout-not-independent-external-corpus",
        "gate_version":report["version"],"threshold":report["threshold"],
        "sample_count":len(rows),
        "coverage":fraction(len(accepted),len(rows)),
        "review_rate":fraction(len(rows)-len(accepted),len(rows)),
        "selective_accuracy":fraction(sum(r["correct"] for r in accepted),len(accepted)),
        "strict_accuracy":fraction(sum(r["correct"] for r in rows),len(rows)),
        "known_recall":fraction(sum(r["correct"] for r in known),len(known)),
        "unknown_reject_rate":fraction(sum(r["correct"] for r in unknown),len(unknown)),
        "unknown_false_accept_rate":fraction(len(false_known),len(unknown)),
        "false_known_count":len(false_known),
        "by_category":by_category,
        "decisions":dict(Counter(r["decision"] for r in rows)),
        "results":rows
    }

def main():
    p=argparse.ArgumentParser(description="v0.9.3 frozen separate-data evaluation")
    p.add_argument("cases");p.add_argument("models");p.add_argument("output")
    p.add_argument("--device",choices=["auto","cpu","cuda"],default="auto")
    p.add_argument("--threshold",type=float,default=0.85)
    a=p.parse_args()
    cases=json.loads(Path(a.cases).read_text(encoding="utf-8-sig"))
    models=json.loads(Path(a.models).read_text(encoding="utf-8-sig"))
    result=summarize(evaluate_gate(cases,models,threshold=a.threshold,device=a.device))
    out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({k:v for k,v in result.items() if k!="results"},ensure_ascii=False,indent=2))

if __name__=="__main__":main()
