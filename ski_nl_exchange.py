"""SEMOS_S2N v0.5: natural-language SKI exchange between heterogeneous LLMs.

Transport-neutral prototype. Each receiver locally encodes Japanese X as its
own SD; no shared embedding coordinate system is assumed. No remote networking
or LLM inference is performed in this module.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
from rsd import make_sd, validate, RSDIntegrityError

FORMAT = "ski-natural-language-v0.5"

def sha(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

def export_message(sd, *, source_llm, target_llm, parent_sd_id=None):
    text = validate(sd)
    if not source_llm or not target_llm or not isinstance(source_llm, str) or not isinstance(target_llm, str):
        raise ValueError("source_llm and target_llm must be nonempty strings")
    return {
        "format": FORMAT,
        "message_id": "ski-" + sha(source_llm + "\x00" + target_llm + "\x00" + sd["sd_id"] + "\x00" + text)[:24],
        "source_llm": source_llm,
        "target_llm": target_llm,
        "source_sd_id": sd["sd_id"],
        "parent_sd_id": parent_sd_id,
        "language": "ja",
        "text": text,
        "text_sha256": sha(text),
        "derivation": sd.get("derivation")
    }

def import_message(message, *, receiver_llm):
    if not isinstance(message, dict) or message.get("format") != FORMAT:
        raise ValueError("invalid SKI message format")
    if not isinstance(receiver_llm, str) or not receiver_llm or message.get("target_llm") != receiver_llm:
        raise ValueError("recipient LLM mismatch")
    for key in ("source_llm", "source_sd_id", "message_id"):
        if not isinstance(message.get(key), str) or not message[key]:
            raise ValueError("missing " + key)
    if message.get("language") != "ja" or not isinstance(message.get("text"), str):
        raise ValueError("invalid natural language text")
    if sha(message["text"]) != message.get("text_sha256"):
        raise RSDIntegrityError("transmitted Japanese text hash mismatch")
    expected = "ski-" + sha(message["source_llm"] + "\x00" + receiver_llm + "\x00" + message["source_sd_id"] + "\x00" + message["text"])[:24]
    if expected != message["message_id"]:
        raise RSDIntegrityError("message identity mismatch")
    if message.get("parent_sd_id") is not None and not isinstance(message["parent_sd_id"], str):
        raise ValueError("invalid parent ID")
    # This mock adapter is intentionally reversible: actual receiver LLM
    # re-semantification should replace this encode step in a later version.
    destination = make_sd(message["text"])
    destination["receiver_llm"] = receiver_llm
    destination["origin"] = {
        "source_llm": message["source_llm"],
        "source_sd_id": message["source_sd_id"],
        "parent_sd_id": message.get("parent_sd_id"),
        "message_id": message["message_id"],
        "derivation": message.get("derivation")
    }
    return destination

def receipt(destination_sd, *, result_text=None):
    text = validate(destination_sd)
    origin = destination_sd.get("origin")
    if not isinstance(origin, dict) or not origin.get("message_id"):
        raise ValueError("SD does not have exchange provenance")
    result = text if result_text is None else result_text
    if not isinstance(result, str):
        raise ValueError("result_text must be a string")
    return {
        "format": FORMAT + "-result",
        "correlation_id": origin["message_id"],
        "source_llm": destination_sd["receiver_llm"],
        "target_llm": origin["source_llm"],
        "source_sd_id": origin["source_sd_id"],
        "text": result,
        "text_sha256": sha(result)
    }

def validate_receipt(message, result):
    if not isinstance(result, dict) or result.get("format") != FORMAT + "-result":
        raise ValueError("invalid receipt")
    if result.get("correlation_id") != message.get("message_id") or result.get("target_llm") != message.get("source_llm") or result.get("source_sd_id") != message.get("source_sd_id") or result.get("source_llm") != message.get("target_llm"):
        raise RSDIntegrityError("receipt routing mismatch")
    if not isinstance(result.get("text"), str) or sha(result["text"]) != result.get("text_sha256"):
        raise RSDIntegrityError("receipt content mismatch")
    return result["text"]

def main(argv=None):
    parser = argparse.ArgumentParser(description="v0.5 SKI natural-language exchange prototype")
    sub = parser.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("export")
    e.add_argument("sd_json")
    e.add_argument("output")
    e.add_argument("--source", required=True)
    e.add_argument("--target", required=True)
    i = sub.add_parser("import")
    i.add_argument("message_json")
    i.add_argument("output")
    i.add_argument("--receiver", required=True)
    r = sub.add_parser("receipt")
    r.add_argument("destination_sd_json")
    r.add_argument("output")
    r.add_argument("--text", default=None)
    args = parser.parse_args(argv)
    try:
        obj = json.loads(Path(getattr(args, "sd_json", getattr(args, "message_json", getattr(args, "destination_sd_json", "")))).read_text(encoding="utf-8-sig"))
        if args.cmd == "export":
            data = export_message(obj, source_llm=args.source, target_llm=args.target,
                                  parent_sd_id=(obj.get("derivation") or {}).get("parent_id"))
        elif args.cmd == "import":
            data = import_message(obj, receiver_llm=args.receiver)
        else:
            data = receipt(obj, result_text=args.text)
        Path(args.output).write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(data, ensure_ascii=False, indent=2))
    except (OSError, ValueError, TypeError, KeyError) as exc:
        parser.exit(2, f"error: {exc}\n")

if __name__ == "__main__":
    main()
