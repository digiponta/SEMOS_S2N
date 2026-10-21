"""RSD v0.3 experiments: lossless compression and deep split-tree verification.

The semantic structure is unchanged from rsd-0.2. Compression is an optional
storage transport envelope; it does not recover meaning from a vector.
"""
from __future__ import annotations
import argparse
import base64
import copy
import json
import random
import zlib
from pathlib import Path
from rsd import RSDIntegrityError, digest, make_sd, merge, split, validate

PACK_FORMAT = "rsd-zlib-v0.3"

def packed_size(obj: dict) -> int:
    return len(json.dumps(obj, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))

def pack(sd: dict, level: int = 9) -> dict:
    text = validate(sd)
    if not isinstance(level, int) or not 0 <= level <= 9:
        raise ValueError("compression level must be 0..9")
    compressed = zlib.compress(text.encode("utf-8"), level=level)
    result = {key: copy.deepcopy(value) for key, value in sd.items() if key != "payload"}
    result["transport"] = {
        "format": PACK_FORMAT,
        "codec": "utf-8+zlib+base64",
        "byte_length": len(text.encode("utf-8")),
        "data": base64.b64encode(compressed).decode("ascii")
    }
    return result

def unpack(obj: dict, *, max_bytes: int = 16_777_216) -> dict:
    if not isinstance(obj, dict) or obj.get("sd_version") != "rsd-0.2":
        raise RSDIntegrityError("not a supported RSD record")
    transport = obj.get("transport")
    if not isinstance(transport, dict) or transport.get("format") != PACK_FORMAT or transport.get("codec") != "utf-8+zlib+base64":
        raise RSDIntegrityError("unsupported transport")
    size = transport.get("byte_length")
    if type(size) is not int or not 0 <= size <= max_bytes:
        raise RSDIntegrityError("invalid or excessive uncompressed length")
    encoded = transport.get("data")
    if not isinstance(encoded, str):
        raise RSDIntegrityError("missing compressed data")
    try:
        stream = base64.b64decode(encoded, validate=True)
        decompressor = zlib.decompressobj()
        output = decompressor.decompress(stream, max_bytes + 1)
        if len(output) > max_bytes or decompressor.unconsumed_tail or not decompressor.eof or decompressor.unused_data:
            raise RSDIntegrityError("invalid or oversized compressed stream")
        output += decompressor.flush(max_bytes + 1 - len(output))
        if len(output) != size:
            raise RSDIntegrityError("decompressed length mismatch")
        text = output.decode("utf-8")
    except (ValueError, UnicodeError, zlib.error) as exc:
        raise RSDIntegrityError("malformed compressed SD") from exc
    result = {key: copy.deepcopy(value) for key, value in obj.items() if key != "transport"}
    result["payload"] = text
    validate(result)
    return result

def deep_split(sd: dict, depth: int) -> list[dict]:
    """Recursively bisect each nontrivial fragment, preserving parent links."""
    if type(depth) is not int or depth < 0:
        raise ValueError("depth must be nonnegative integer")
    text = validate(sd)
    if depth == 0 or len(text) < 2:
        return [copy.deepcopy(sd)]
    left, right = split(sd, len(text) // 2)
    return deep_split(left, depth - 1) + deep_split(right, depth - 1)

def reconstruct_leaves(leaves: list[dict]) -> dict:
    """Rebuild one hierarchy bottom-up using only leaf records."""
    if not leaves:
        raise RSDIntegrityError("missing all leaves")
    current = {}
    for leaf in leaves:
        validate(leaf)
        identity = leaf["sd_id"]
        if identity in current:
            raise RSDIntegrityError("duplicate leaf")
        current[identity] = copy.deepcopy(leaf)
    while True:
        if len(current) == 1:
            single = next(iter(current.values()))
            if "reconstruction" not in single:
                return single
            raise RSDIntegrityError("incomplete tree or missing sibling")
        candidates = {}
        for item in current.values():
            meta = item.get("reconstruction")
            if isinstance(meta, dict):
                candidates.setdefault(meta.get("parent_id"), []).append(item)
        progressed = False
        for parent_id, parts in candidates.items():
            if len(parts) == 2:
                parent = merge(parts)
                for item in parts:
                    del current[item["sd_id"]]
                if parent_id in current:
                    raise RSDIntegrityError("duplicate reconstructed parent")
                current[parent_id] = parent
                progressed = True
                break
        if not progressed:
            raise RSDIntegrityError("missing parts or incompatible hierarchy")

def benchmark(text: str, depth: int = 3, seed: int = 42) -> dict:
    parent = make_sd(text)
    leaves = deep_split(parent, depth)
    shuffled = list(leaves)
    random.Random(seed).shuffle(shuffled)
    reconstructed = reconstruct_leaves(shuffled)
    compressed = pack(parent)
    packed_restored = unpack(compressed)
    raw_bytes = len(text.encode("utf-8"))
    sd_bytes = packed_size(parent)
    packed_bytes = packed_size(compressed)
    leaf_bytes = sum(packed_size(leaf) for leaf in leaves)
    return {
        "depth": depth,
        "leaves": len(leaves),
        "exact_reconstruction": reconstructed == parent == packed_restored,
        "raw_utf8_bytes": raw_bytes,
        "rsd_json_bytes": sd_bytes,
        "zlib_envelope_bytes": packed_bytes,
        "all_leaf_json_bytes": leaf_bytes,
        "packed_to_plain_ratio": round(packed_bytes / sd_bytes, 4) if sd_bytes else None
    }

def main():
    parser = argparse.ArgumentParser(description="RSD v0.3 lossless compression and hierarchical benchmark")
    parser.add_argument("input", help="UTF-8 Japanese text file")
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--output", default="results/rsd_v03_benchmark.json")
    args = parser.parse_args()
    try:
        text = Path(args.input).read_text(encoding="utf-8-sig")
        result = benchmark(text, args.depth)
        target = Path(args.output)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(result, ensure_ascii=False, indent=2))
    except (RSDIntegrityError, OSError, ValueError) as exc:
        parser.exit(2, f"error: {exc}\n")

if __name__ == "__main__":
    main()
