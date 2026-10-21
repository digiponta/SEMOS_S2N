"""SEMOS_S2N v0.4: AI-derived propositions with independently reversible SDs.

A and propositions B/C are NOT string slices. Derivation is supplied by an
external AI or researcher; this module validates and records it, but does not
claim to infer semantic equivalence automatically.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from rsd import make_sd, validate, digest, RSDIntegrityError

FORMAT = "sd-proposition-v0.4"

def derive(source: str, propositions: list[str], *, relation: str = "conjunction", model_id: str = "external-ai") -> list[dict]:
    if not isinstance(source, str) or not source:
        raise ValueError("source Japanese A is required")
    if not isinstance(propositions, list) or len(propositions) < 2 or any(not isinstance(p, str) or not p for p in propositions):
        raise ValueError("at least two nonempty Japanese propositions are required")
    if relation not in ("conjunction", "explanation", "sequence", "unspecified"):
        raise ValueError("unsupported proposition relation")
    if not isinstance(model_id, str) or not model_id:
        raise ValueError("model_id required")
    parent = make_sd(source)
    parent_id = parent["sd_id"]
    outputs = []
    # The original A is included as a lossless reconstruction witness. Without
    # it (or equivalent residual bits), novel B and C cannot guarantee A.
    for index, sentence in enumerate(propositions):
        child = make_sd(sentence, f"{parent_id}/derived-{index}")
        child["derivation"] = {
            "format": FORMAT,
            "parent_id": parent_id,
            "parent_hash": digest(source),
            "proposition_count": len(propositions),
            "index": index,
            "relation": relation,
            "model_id": model_id,
            "source_witness": source,
            "siblings": [f"{parent_id}/derived-{j}" for j in range(len(propositions)) if j != index],
            "semantic_equivalence": "unverified"
        }
        outputs.append(child)
    return outputs

def assemble(parts: list[dict]) -> dict:
    if not isinstance(parts, list) or len(parts) < 2:
        raise RSDIntegrityError("missing propositions")
    metas = []
    for part in parts:
        validate(part)
        meta = part.get("derivation")
        if not isinstance(meta, dict) or meta.get("format") != FORMAT:
            raise RSDIntegrityError("missing derived-proposition metadata")
        metas.append(meta)
    keys = ("parent_id", "parent_hash", "proposition_count", "relation", "model_id", "source_witness", "semantic_equivalence")
    reference = metas[0]
    if any(any(meta.get(key) != reference.get(key) for key in keys) for meta in metas):
        raise RSDIntegrityError("inconsistent derivation metadata")
    count = reference.get("proposition_count")
    indices = [m.get("index") for m in metas]
    if type(count) is not int or count < 2 or len(parts) != count or any(type(i) is not int for i in indices) or sorted(indices) != list(range(count)):
        raise RSDIntegrityError("missing or duplicated propositions")
    ids = [f"{reference['parent_id']}/derived-{i}" for i in range(count)]
    for part, meta in zip(parts, metas):
        if part["sd_id"] != ids[meta["index"]] or meta.get("siblings") != [name for name in ids if name != part["sd_id"]]:
            raise RSDIntegrityError("inconsistent proposition identity or siblings")
    witness = reference.get("source_witness")
    if not isinstance(witness, str) or digest(witness) != reference.get("parent_hash"):
        raise RSDIntegrityError("invalid source reconstruction witness")
    parent = make_sd(witness, reference["parent_id"])
    parent["derivation_summary"] = {
        "format": FORMAT, "relation": reference["relation"],
        "model_id": reference["model_id"],
        "proposition_ids": ids,
        "semantic_equivalence": "unverified"
    }
    return parent

def main(argv=None):
    parser = argparse.ArgumentParser(description="AI-proposition SD derivation and reconstruction")
    sub = parser.add_subparsers(dest="command", required=True)
    a = sub.add_parser("derive")
    a.add_argument("input", help="JSON with source, propositions, relation (optional), model_id (optional)")
    a.add_argument("output", help="output JSON file containing SD(B), SD(C), ...")
    b = sub.add_parser("assemble")
    b.add_argument("input", help="JSON file with list of derived SD objects")
    b.add_argument("output", help="output reconstructed parent SD JSON")
    c = sub.add_parser("decode")
    c.add_argument("input", help="one derived SD JSON")
    args = parser.parse_args(argv)
    try:
        if args.command == "derive":
            obj = json.loads(Path(args.input).read_text(encoding="utf-8-sig"))
            outputs = derive(obj["source"], obj["propositions"], relation=obj.get("relation", "conjunction"), model_id=obj.get("model_id", "external-ai"))
            Path(args.output).write_text(json.dumps(outputs, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
        elif args.command == "assemble":
            parts = json.loads(Path(args.input).read_text(encoding="utf-8-sig"))
            parent = assemble(parts)
            Path(args.output).write_text(json.dumps(parent, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
            print(validate(parent))
        else:
            print(validate(json.loads(Path(args.input).read_text(encoding="utf-8-sig"))))
    except (ValueError, TypeError, KeyError, OSError) as e:
        parser.exit(2, f"error: {e}\n")

if __name__ == "__main__":
    main()
