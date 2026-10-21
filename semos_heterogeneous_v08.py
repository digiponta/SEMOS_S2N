"""v0.8 cross-checkpoint semantic interchange evaluation.

Vectors are kept model-local. This report evaluates transport fidelity and
distinct model fingerprints, not cross-model cosine similarity or entailment.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from rsd import validate
from ski_nl_exchange import validate_receipt
from semos_real_adapter_v07 import run_messages

def evaluate(messages, models, *, device="auto", require_distinct=True):
    result = run_messages(messages, models, device=device)
    fingerprints = []
    rows = []
    for message, item in zip(messages, result["results"]):
        local = item["local_sd"]
        rep = local["receiver_semantic"]
        fingerprint = (rep["encoder_sha256"], rep["tokenizer_sha256"])
        fingerprints.append(fingerprint)
        input_text = message["text"]
        decoded = validate(local)
        reply_text = validate_receipt(message, item["response"])
        rows.append({
            "receiver": item["receiver"],
            "source_sd_id": message["source_sd_id"],
            "encoder_sha256": fingerprint[0],
            "tokenizer_sha256": fingerprint[1],
            "semantic_dimension": rep["dimension"],
            "token_count": rep["token_count"],
            "transport_text_exact": input_text == decoded,
            "reply_text_exact": input_text == reply_text,
            "vector_norm": round(sum(float(x)**2 for x in rep["vector"])**0.5, 8),
        })
    distinct = len(set(fingerprints)) == len(fingerprints)
    if require_distinct and not distinct:
        raise ValueError("distinct receiver model fingerprints required; configure different checkpoints")
    return {
        "version": "v0.8",
        "model_count": len(rows),
        "distinct_model_fingerprints": distinct,
        "all_transport_exact": all(x["transport_text_exact"] for x in rows),
        "all_replies_exact": all(x["reply_text_exact"] for x in rows),
        "semantic_equivalence": "not-evaluated",
        "cross_model_vector_comparison": "not-applicable-without-common-vector-basis",
        "real_inference": "not-implemented",
        "results": rows
    }

def main():
    p=argparse.ArgumentParser(description="SEMOS v0.8 cross-checkpoint exchange audit")
    p.add_argument("messages")
    p.add_argument("models")
    p.add_argument("output")
    p.add_argument("--device",choices=["auto","cpu","cuda"],default="auto")
    p.add_argument("--allow-identical",action="store_true",help="smoke test only; not heterogeneous")
    a=p.parse_args()
    try:
        messages=json.loads(Path(a.messages).read_text(encoding="utf-8-sig"))
        models=json.loads(Path(a.models).read_text(encoding="utf-8-sig"))
        result=evaluate(messages,models,device=a.device,require_distinct=not a.allow_identical)
        dest=Path(a.output)
        dest.parent.mkdir(parents=True,exist_ok=True)
        dest.write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
        print(json.dumps(result,ensure_ascii=False,indent=2))
    except (OSError, ValueError, TypeError, KeyError, RuntimeError) as exc:
        p.exit(2,f"error: {exc}\n")

if __name__=="__main__":
    main()
