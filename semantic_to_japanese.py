"""Safe Japanese renderer for LLM_SEM SemanticData and propositions."""
import argparse
import json
import sys
from pathlib import Path


class ConversionError(ValueError):
    pass


def convert(data):
    """Render source text or explicit symbolic data; never infer from vectors."""
    if isinstance(data, list):
        if not data:
            raise ConversionError("empty list")
        return "\n".join(convert(item) for item in data)
    if not isinstance(data, dict):
        raise ConversionError("expected object or array")
    if "propositions" in data:
        return convert(data["propositions"])
    if isinstance(data.get("text"), str) and data["text"].strip():
        return data["text"].strip()
    s, v = data.get("subject"), data.get("value", data.get("object"))
    if isinstance(s, str) and s.strip() and isinstance(v, str) and v.strip():
        s, v = s.strip().rstrip("。"), v.strip().rstrip("。")
        p = data.get("predicate", data.get("relation", "is"))
        if p in ("is", "be", "である", None):
            return f"{s}は、{v}である。"
        if p in ("includes", "include", "含む"):
            return f"{s}は、{v}を含む。"
        if p in ("has", "have", "持つ"):
            return f"{s}は、{v}を持つ。"
        raise ConversionError(f"unsupported predicate: {p}")
    if "vector" in data:
        raise ConversionError("vector-only SemanticData requires a compatible trained decoder or source text")
    raise ConversionError("unsupported SemanticData structure")


def main():
    p = argparse.ArgumentParser(description="Convert LLM_SEM Semantic Data to Japanese")
    p.add_argument("input", nargs="?", default="-", help="JSON or JSONL file; '-' for stdin")
    p.add_argument("-o", "--output")
    p.add_argument("--jsonl", action="store_true")
    args = p.parse_args()
    try:
        raw = sys.stdin.read() if args.input == "-" else Path(args.input).read_text(encoding="utf-8-sig")
    except FileNotFoundError:
        print(f"input file not found: {args.input}. Try: python semantic_to_japanese.py examples/semantic.json", file=sys.stderr)
        return 2
    except OSError as e:
        print(f"input read error: {e}", file=sys.stderr)
        return 2
    try:
        data = [json.loads(s) for s in raw.splitlines() if s.strip()] if args.jsonl else json.loads(raw)
        result = convert(data)
    except (ValueError, ConversionError) as e:
        print(f"conversion error: {e}", file=sys.stderr)
        return 2
    if args.output:
        Path(args.output).write_text(result + "\n", encoding="utf-8")
    else:
        print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
