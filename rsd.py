"""Lossless Reconstructable Semantic Data (RSD) v0.2.

Exact Unicode round trips; split metadata is duplicated on each child so
parts can be moved independently. This format does not claim semantic
understanding from a 64-D vector.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

VERSION = "rsd-0.2"
class RSDIntegrityError(ValueError):
    pass

def digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

def make_sd(text: str, sd_id: str | None = None) -> dict:
    if not isinstance(text, str):
        raise TypeError("text must be str")
    h = digest(text)
    return {"sd_version": VERSION, "sd_id": sd_id or f"sd-{h[:24]}",
            "payload": text, "text_sha256": h}

def validate(sd: dict) -> str:
    if not isinstance(sd, dict) or sd.get("sd_version") != VERSION:
        raise RSDIntegrityError("unsupported or missing SD version")
    payload = sd.get("payload")
    if not isinstance(payload, str) or digest(payload) != sd.get("text_sha256"):
        raise RSDIntegrityError("payload checksum mismatch")
    if not isinstance(sd.get("sd_id"), str) or not sd["sd_id"]:
        raise RSDIntegrityError("missing sd_id")
    return payload

def split(sd: dict, position: int) -> tuple[dict, dict]:
    text = validate(sd)
    if not isinstance(position, int) or isinstance(position, bool) or not 0 < position < len(text):
        raise ValueError("position must be strictly inside the Unicode text")
    pieces = (text[:position], text[position:])
    manifest = {
        "parent_id": sd["sd_id"],
        "parent_hash": sd["text_sha256"],
        "part_count": 2,
        "join_operator": "concat",
        "total_length": len(text),
        "parent_reconstruction": sd.get("reconstruction")
    }
    result = []
    for i, chunk in enumerate(pieces):
        child = make_sd(chunk, f"{sd['sd_id']}/part-{i}")
        child["reconstruction"] = {**manifest, "part_index": i}
        result.append(child)
    return result[0], result[1]

def merge(parts: list[dict]) -> dict:
    if not isinstance(parts, list) or not parts:
        raise RSDIntegrityError("no parts to merge")
    rows = []
    for sd in parts:
        text = validate(sd)
        meta = sd.get("reconstruction")
        if not isinstance(meta, dict):
            raise RSDIntegrityError("fragment missing reconstruction metadata")
        rows.append((sd, text, meta))
    reference = rows[0][2]
    for field in ("parent_id", "parent_hash", "part_count", "join_operator", "total_length", "parent_reconstruction"):
        if any(row[2].get(field) != reference.get(field) for row in rows):
            raise RSDIntegrityError(f"mismatched {field}")
    count = reference.get("part_count")
    if not isinstance(count, int) or isinstance(count, bool) or count != 2 or len(rows) != count:
        raise RSDIntegrityError("missing or extra fragments")
    if reference.get("join_operator") != "concat":
        raise RSDIntegrityError("unsupported join operator")
    indices = [row[2].get("part_index") for row in rows]
    if any(type(i) is not int for i in indices) or sorted(indices) != list(range(count)):
        raise RSDIntegrityError("duplicate or invalid fragment positions")
    if any(row[0]["sd_id"] != f"{reference['parent_id']}/part-{row[2]['part_index']}" for row in rows):
        raise RSDIntegrityError("fragment identity mismatch")
    full = "".join(text for _, text, meta in sorted(rows, key=lambda x: x[2]["part_index"]))
    if len(full) != reference.get("total_length") or digest(full) != reference.get("parent_hash"):
        raise RSDIntegrityError("parent reconstruction mismatch")
    reconstructed = make_sd(full, reference["parent_id"])
    if reconstructed["text_sha256"] != reference["parent_hash"]:
        raise RSDIntegrityError("parent hash mismatch")
    ancestor = reference.get("parent_reconstruction")
    if ancestor is not None:
        if not isinstance(ancestor, dict):
            raise RSDIntegrityError("invalid ancestor reconstruction metadata")
        reconstructed["reconstruction"] = ancestor
    return reconstructed

def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))

def write(path, obj):
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

def main(argv=None):
    p = argparse.ArgumentParser(description="Lossless SD encode/split/merge/decode")
    sub = p.add_subparsers(dest="action", required=True)
    a = sub.add_parser("encode")
    a.add_argument("input", help="UTF-8 original Japanese text file")
    a.add_argument("output")
    b = sub.add_parser("split")
    b.add_argument("input")
    b.add_argument("position", type=int, help="Unicode code-point split index")
    b.add_argument("left")
    b.add_argument("right")
    c = sub.add_parser("merge")
    c.add_argument("left")
    c.add_argument("right")
    c.add_argument("output")
    d = sub.add_parser("decode")
    d.add_argument("input")
    d.add_argument("output", help="Original Japanese text output")
    args = p.parse_args(argv)
    try:
        if args.action == "encode":
            write(args.output, make_sd(Path(args.input).read_text(encoding="utf-8-sig")))
        elif args.action == "split":
            a, b = split(read(args.input), args.position)
            write(args.left, a)
            write(args.right, b)
        elif args.action == "merge":
            write(args.output, merge([read(args.left), read(args.right)]))
        else:
            text = validate(read(args.input))
            Path(args.output).write_text(text, encoding="utf-8")
    except (OSError, ValueError, TypeError) as exc:
        p.exit(2, f"error: {exc}\n")

if __name__ == "__main__":
    main()
