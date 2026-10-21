"""SEMOS_S2N v0.6: pluggable heterogeneous-LLM dispatch and aggregation.

No network or external LLM call is claimed. The callback interface is where a
real receiver model's Japanese -> local Semantic Data -> result runs.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from rsd import make_sd, validate, RSDIntegrityError
from ski_nl_exchange import export_message, import_message, receipt, validate_receipt

FORMAT = "semos-dispatch-v0.6"

def process(message: dict, receiver: str, adapter=None) -> tuple[dict, dict]:
    """Adapter(text, model_id) returns an SD dict or Japanese result string."""
    local_sd = import_message(message, receiver_llm=receiver)
    if adapter is None:
        result_text = validate(local_sd)  # Explicit simulation mode.
        mode = "echo-simulator"
    else:
        value = adapter(validate(local_sd), receiver)
        if isinstance(value, dict):
            result_text = validate(value)
        elif isinstance(value, str):
            result_text = value
        else:
            raise TypeError("adapter must return Japanese string or valid SD")
        mode = "external-adapter"
    answer = receipt(local_sd, result_text=result_text)
    answer["processing_mode"] = mode
    return local_sd, answer

def run(parts: list[dict], receivers: list[str], *, sender: str = "LLM-A", adapters=None) -> dict:
    if not isinstance(parts, list) or not isinstance(receivers, list) or len(parts) < 2 or len(parts) != len(receivers):
        raise ValueError("parts and receivers must have equal size >= 2")
    if len(set(receivers)) != len(receivers) or sender in receivers or any(not isinstance(r, str) or not r for r in receivers):
        raise ValueError("receiver IDs must be unique, nonempty and distinct from sender")
    if not isinstance(sender, str) or not sender:
        raise ValueError("sender ID required")
    adapters = adapters or {}
    if not isinstance(adapters, dict):
        raise ValueError("adapters must be a dictionary")
    records = []
    parent_ids = set()
    for sd, receiver in zip(parts, receivers):
        source_text = validate(sd)
        derivation = sd.get("derivation")
        if not isinstance(derivation, dict) or not isinstance(derivation.get("parent_id"), str):
            raise RSDIntegrityError("derived SD with parent ID required")
        parent_ids.add(derivation["parent_id"])
        # Send only Japanese X plus necessary routing metadata.
        # Do not send v0.4 source_witness or unrelated propositions.
        portable = make_sd(source_text, sd["sd_id"])
        message = export_message(portable, source_llm=sender, target_llm=receiver, parent_sd_id=derivation["parent_id"])
        local, answer = process(message, receiver, adapters.get(receiver))
        verified = validate_receipt(message, answer)
        records.append({
            "receiver": receiver, "source_sd_id": sd["sd_id"],
            "message": message, "receiver_sd": local,
            "result": answer, "verified_result_text": verified,
        })
    if len(parent_ids) != 1:
        raise RSDIntegrityError("mixed parents")
    return {
        "format": FORMAT,
        "sender": sender,
        "parent_sd_id": next(iter(parent_ids)),
        "all_receipts_valid": True,
        "semantic_equivalence": "not-evaluated",
        "mode": "adapter" if adapters else "echo-simulator",
        "results": records,
    }

def main(argv=None):
    parser = argparse.ArgumentParser(description="SEMOS v0.6 local multi-LLM routing simulation")
    parser.add_argument("derived_sd_json")
    parser.add_argument("output")
    parser.add_argument("--receivers", nargs="+", default=["LLM-B", "LLM-C"])
    parser.add_argument("--sender", default="LLM-A")
    args = parser.parse_args(argv)
    try:
        parts = json.loads(Path(args.derived_sd_json).read_text(encoding="utf-8-sig"))
        result = run(parts, args.receivers, sender=args.sender)
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({
            "format": FORMAT, "mode": result["mode"],
            "all_receipts_valid": result["all_receipts_valid"],
            "semantic_equivalence": result["semantic_equivalence"],
            "results": [{ "receiver": row["receiver"], "text": row["verified_result_text"] } for row in result["results"]]
        }, ensure_ascii=False, indent=2))
    except (OSError, ValueError, TypeError, KeyError) as exc:
        parser.exit(2, f"error: {exc}\n")

if __name__ == "__main__":
    main()
