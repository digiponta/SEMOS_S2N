"""SEMOS_S2N v0.9: model-local Semantic Memory and Known/Unknown evaluation.

Cosine retrieval is only performed within one encoder checkpoint + tokenizer.
This is a local research baseline, not language entailment or model inference.
"""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path
from rsd import digest
from semos_real_adapter_v07 import encode_japanese

def cosine(a,b):
    if len(a)!=len(b) or not a or any(not math.isfinite(float(x)) for x in a+b):
        raise ValueError("vectors must have matching finite dimensions")
    na=math.sqrt(sum(x*x for x in a)); nb=math.sqrt(sum(x*x for x in b))
    if na<=0 or nb<=0:
        raise ValueError("zero-length vector norm")
    return sum(x*y for x,y in zip(a,b))/(na*nb)

class LocalMemory:
    def __init__(self,model_id,threshold=0.85):
        if not model_id or not 0<=threshold<=1: raise ValueError("invalid memory settings")
        self.model_id=model_id
        self.threshold=threshold
        self.records=[]
        self.fingerprint=None

    def insert(self,text,encoded):
        if not isinstance(text,str) or not text:
            raise ValueError("nonempty Japanese text required")
        vector=encoded["vector"]
        if encoded.get("dimension")!=len(vector) or len(vector)!=64:
            raise ValueError("expected 64D receiver vector")
        ident=(encoded["encoder_sha256"],encoded["tokenizer_sha256"])
        cosine(vector,vector)
        if self.fingerprint is None:
            self.fingerprint=ident
        if ident!=self.fingerprint:
            raise ValueError("mixed encoder/tokenizer fingerprints in one memory")
        if any(record["text_sha256"]==digest(text) for record in self.records):
            return False
        self.records.append({"text":text,"text_sha256":digest(text),"vector":list(vector)})
        return True

    def query(self,text,encoded):
        ident=(encoded["encoder_sha256"],encoded["tokenizer_sha256"])
        if self.fingerprint is not None and ident!=self.fingerprint:
            raise ValueError("query encoder does not match receiver memory")
        if not self.records:
            return {"decision":"UNKNOWN","reason":"empty-memory","score":None,"match":None}
        similarities=[cosine(encoded["vector"],r["vector"]) for r in self.records]
        best=max(range(len(similarities)),key=similarities.__getitem__)
        score=similarities[best]
        known=score>=self.threshold
        return {"decision":"KNOWN" if known else "UNKNOWN","reason":"similarity-threshold",
                "score":round(score,6),"match":self.records[best]["text"] if known else None}

def evaluate(examples,models,threshold=0.85,device="auto"):
    if not isinstance(examples,list) or not examples:
        raise ValueError("examples required")
    memories={}
    output=[]
    for row in examples:
        receiver=row["receiver"]
        spec=models[receiver]
        if receiver not in memories:
            memories[receiver]=LocalMemory(receiver,threshold)
        text=row["text"]
        encoded=encode_japanese(text,checkpoint=spec["checkpoint"],tokenizer_path=spec["tokenizer"],device=device)
        if row["operation"]=="teach":
            inserted=memories[receiver].insert(text,encoded)
            output.append({"receiver":receiver,"operation":"teach","inserted":inserted,"text":text})
        elif row["operation"]=="query":
            decision=memories[receiver].query(text,encoded)
            output.append({"receiver":receiver,"operation":"query","text":text,**decision,
                           "expected":row.get("expected")})
        else:
            raise ValueError("operation must be teach or query")
    checks=[r for r in output if r["operation"]=="query" and r["expected"] is not None]
    accuracy=(sum(r["decision"]==r["expected"] for r in checks)/len(checks)) if checks else None
    return {"version":"v0.9","semantic_equivalence":"not-evaluated",
            "method":"model-local-cosine-gate","threshold":threshold,
            "labeled_query_count":len(checks),"labeled_query_accuracy":accuracy,
            "results":output}

def main():
    p=argparse.ArgumentParser(description="SEMOS v0.9 local memory retrieval benchmark")
    p.add_argument("examples");p.add_argument("models");p.add_argument("output")
    p.add_argument("--threshold",type=float,default=0.85)
    p.add_argument("--device",choices=["auto","cpu","cuda"],default="auto")
    a=p.parse_args()
    try:
        examples=json.loads(Path(a.examples).read_text(encoding="utf-8-sig"))
        models=json.loads(Path(a.models).read_text(encoding="utf-8-sig"))
        result=evaluate(examples,models,threshold=a.threshold,device=a.device)
        path=Path(a.output);path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
        print(json.dumps(result,ensure_ascii=False,indent=2))
    except (OSError,KeyError,ValueError,TypeError) as exc:
        p.exit(2,f"error: {exc}\n")

if __name__=="__main__":main()
