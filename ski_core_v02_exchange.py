"""SKI v0.2 draft Core JSON-LD Dataset A/B validator and local roundtrip.

Based on digiponta/HASKI_DOC SKI v0.2 editor's draft (2026-10-19).
Subset validation, NOT official JSON-LD/SHACL conformance. No network.
"""
import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path

LIFECYCLES={"CANDIDATE","APPROVED","SUPERSEDED","RETRACTED"}
PROFILE="SKI-Core-v0.2-draft"
def valid_iri(value):
    return isinstance(value,str) and (value.startswith("urn:") or value.startswith("https://")) and len(value)>5
def require_iri_ref(value, field):
    if not isinstance(value,dict) or not valid_iri(value.get("@id")):
        raise ValueError(field+" requires an IRI @id")
def validate_assertion(a):
    if not isinstance(a,dict): raise ValueError("assertion must be object")
    if not isinstance(a.get("@context"),dict) or not a["@context"].get("ski"):
        raise ValueError("missing local SKI JSON-LD context mapping")
    if not valid_iri(a.get("@id")): raise ValueError("missing assertion IRI")
    if a.get("@type")!="ski:KnowledgeAssertion": raise ValueError("invalid assertion type")
    if type(a.get("ski:version")) is not int or a["ski:version"]<1:
        raise ValueError("invalid version")
    require_iri_ref(a.get("ski:issuer"),"issuer")
    ts=a.get("ski:recordedAt")
    if not isinstance(ts,str):raise ValueError("invalid recordedAt")
    try: date=datetime.fromisoformat(ts.replace("Z","+00:00"))
    except ValueError: raise ValueError("invalid recordedAt")
    if date.tzinfo is None or date.utcoffset() is None:raise ValueError("recordedAt must include timezone")
    if a.get("ski:lifecycle") not in LIFECYCLES:raise ValueError("invalid lifecycle")
    claim=a.get("ski:claim")
    if not isinstance(claim,dict):raise ValueError("missing claim")
    for field in ("ski:subject","ski:predicate","ski:object"):
        value=claim.get(field)
        if isinstance(value,dict) and "@id" in value:require_iri_ref(value,field)
        elif isinstance(value,dict) and isinstance(value.get("@value"),str):
            if not value["@value"]:raise ValueError("empty literal")
        else: raise ValueError("claim must contain IRI or literal: "+field)
    # Preserve all extra attributes verbatim. The natural-language text
    # property below is a local extension and is not a normative SKI term.
    nl=a.get("s2n:naturalLanguage")
    if nl is not None and (not isinstance(nl,str) or not nl):raise ValueError("invalid natural language")
    return a

def validate_dataset(dataset):
    if not isinstance(dataset,list):raise ValueError("dataset must be JSON-LD assertion list")
    ids=set()
    for row in dataset:
        validate_assertion(row)
        if row["@id"] in ids:raise ValueError("duplicate assertion identifier")
        ids.add(row["@id"])
    return dataset

def canonical_digest(dataset):
    validate_dataset(dataset)
    value=json.dumps(dataset,ensure_ascii=False,sort_keys=True,separators=(",",":"),allow_nan=False)
    return hashlib.sha256(value.encode("utf-8")).hexdigest()

def acknowledge(dataset):
    return {"profile":PROFILE,"assertion_ids":[x["@id"] for x in validate_dataset(dataset)],
            "sha256":canonical_digest(dataset),"status":"RECEIVED"}

def verify_ack(dataset,ack):
    if ack!=acknowledge(dataset):raise ValueError("ACK mismatch")
    return True

def verify_reply(a,b):
    validate_dataset(a);validate_dataset(b)
    for item in b:
        origin=item.get("s2n:inReplyTo")
        if not isinstance(origin,list) or not origin or not all(valid_iri(x) for x in origin):
            raise ValueError("reply must correlate to assertion IRI(s)")
        if not all(x in {row["@id"] for row in a} for x in origin):
            raise ValueError("unmatched reply correlation")
    return True

def main():
    p=argparse.ArgumentParser()
    p.add_argument("dataset_a");p.add_argument("dataset_b");p.add_argument("output")
    args=p.parse_args()
    a=json.loads(Path(args.dataset_a).read_text(encoding="utf-8-sig"))
    b=json.loads(Path(args.dataset_b).read_text(encoding="utf-8-sig"))
    receipt=acknowledge(a);verify_ack(a,receipt);verify_reply(a,b)
    report={"version":"SKI-Core-v0.2-draft","status":"local-protocol-smoke-pass",
            "schema_subset":"draft-subset-not-official-SHACL",
            "dataset_a_count":len(a),"dataset_b_count":len(b),
            "same_core_schema":True,"ack":receipt,"reply_digest":canonical_digest(b)}
    out=Path(args.output);out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(report,ensure_ascii=False,indent=2))
if __name__=="__main__":main()
