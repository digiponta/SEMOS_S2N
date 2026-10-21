"""SKI v0.5 message-array roundtrip; wire datasets A and B share same format.

No semantic interpretation. A dataset is a JSON array of existing
ski-natural-language-v0.5 messages, not a new S2N envelope.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from ski_nl_exchange import FORMAT, export_message, import_message, sha
from rsd import make_sd

def validate_dataset(messages, receiver):
    if not isinstance(messages,list) or not messages:
        raise ValueError("SKI dataset must be a nonempty message array")
    seen=set()
    for m in messages:
        if not isinstance(m,dict) or m.get("format")!=FORMAT:
            raise ValueError("invalid SKI message")
        if m.get("message_id") in seen:
            raise ValueError("duplicate SKI message ID")
        seen.add(m["message_id"])
        import_message(m,receiver_llm=receiver)
    return messages

def digest_dataset(messages):
    return sha(json.dumps(messages,sort_keys=True,separators=(",",":"),ensure_ascii=False))

def ack(messages, receiver):
    validate_dataset(messages,receiver)
    return {"protocol":"ski-receipt-ack-v0.5",
            "receiver_llm":receiver,
            "message_ids":[m["message_id"] for m in messages],
            "dataset_sha256":digest_dataset(messages)}

def verify_ack(messages,receipt,receiver):
    expected=ack(messages,receiver)
    if receipt!=expected:
        raise ValueError("SKI ACK does not match dataset, recipient, or digest")
    return True

def build_reply(messages,receipt,receiver,reply_texts):
    """Local adapter example only; reply texts supplied by caller, not inferred."""
    verify_ack(messages,receipt,receiver)
    if not isinstance(reply_texts,list) or len(reply_texts)!=len(messages):
        raise ValueError("reply text count must match request count in demo adapter")
    replies=[]
    for request,text in zip(messages,reply_texts):
        if not isinstance(text,str):
            raise ValueError("reply must be text")
        sd=make_sd(text)
        response=export_message(sd,source_llm=receiver,target_llm=request["source_llm"],
                                parent_sd_id=request["source_sd_id"])
        # 'derivation' is an existing optional SKI metadata field; do not
        # introduce a new top-level format or ad-hoc transport envelope.
        response["derivation"]={"response_to_message_id":request["message_id"],
                                "source_sd_id":request["source_sd_id"]}
        replies.append(response)
    validate_dataset(replies,messages[0]["source_llm"])
    return replies

def verify_reply(requests,replies,sender):
    validate_dataset(requests,requests[0]["target_llm"])
    validate_dataset(replies,sender)
    if len(requests)!=len(replies):
        raise ValueError("one-to-one demo reply count mismatch")
    pending={m["message_id"]:m for m in requests}
    got=set()
    for item in replies:
        info=item.get("derivation")
        if not isinstance(info,dict):
            raise ValueError("missing response correlation attributes")
        id=info.get("response_to_message_id")
        if id not in pending or id in got:
            raise ValueError("unknown or repeated response correlation ID")
        original=pending[id]
        if item["source_llm"]!=original["target_llm"] or item["target_llm"]!=original["source_llm"]:
            raise ValueError("invalid response routing")
        if info.get("source_sd_id")!=original["source_sd_id"]:
            raise ValueError("invalid source SD linkage")
        got.add(id)
    if got!=set(pending):
        raise ValueError("missing responses")
    return True

def main():
    p=argparse.ArgumentParser(description="Local v0.5 SKI-message dataset roundtrip")
    p.add_argument("dataset_a");p.add_argument("responses");p.add_argument("output")
    p.add_argument("--receiver",required=True)
    a=p.parse_args()
    try:
        messages=json.loads(Path(a.dataset_a).read_text(encoding="utf-8-sig"))
        replies_text=json.loads(Path(a.responses).read_text(encoding="utf-8-sig"))
        receipt=ack(messages,a.receiver)
        verify_ack(messages,receipt,a.receiver)
        replies=build_reply(messages,receipt,a.receiver,replies_text)
        verify_reply(messages,replies,messages[0]["source_llm"])
        result={"status":"local-protocol-smoke-pass","transport":"in-process",
                "ack":receipt,"dataset_b":replies}
        dest=Path(a.output);dest.parent.mkdir(parents=True,exist_ok=True)
        dest.write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
        print(json.dumps({"status":result["status"],"format":FORMAT,
                          "dataset_a_count":len(messages),"dataset_b_count":len(replies),
                          "same_ski_format":all(m["format"]==FORMAT for m in replies),
                          "correlation_verified":True},ensure_ascii=False,indent=2))
    except (OSError,ValueError,KeyError,TypeError) as exc:
        p.exit(2,f"error: {exc}\n")
if __name__=="__main__":main()
