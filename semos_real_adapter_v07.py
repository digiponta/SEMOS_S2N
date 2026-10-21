"""v0.7 real LLM_SEM encoder adapter for SEMOS_S2N.

Loads the user's local PyTorch checkpoint and computes a receiver-local 64-D
semantic vector. No remote LLM inference, training, or memory insertion.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from rsd import make_sd, validate
from ski_nl_exchange import import_message, receipt, validate_receipt

def encode_japanese(text, *, checkpoint, tokenizer_path, device="auto", pooling="hybrid", alpha=0.35, normalized=False):
    import torch
    from semantic_decoder import load_encoder, digest
    if device not in ("auto", "cpu", "cuda"):
        raise ValueError("device must be auto, cpu or cuda")
    chosen = "cuda" if device == "auto" and torch.cuda.is_available() else ("cpu" if device == "auto" else device)
    if chosen == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA device is unavailable")
    checkpoint, tokenizer_path = Path(checkpoint), Path(tokenizer_path)
    if not checkpoint.is_file() or not tokenizer_path.is_file():
        raise FileNotFoundError("model checkpoint/tokenizer missing: see SEMOS_REAL_ADAPTER_V07.md")
    model, tokenizer = load_encoder(checkpoint, tokenizer_path, torch.device(chosen))
    ids = tokenizer.encode(text, add_bos=True, add_eos=True)
    if not ids or len(ids) > model.context_length:
        raise ValueError("token length exceeds encoder context window")
    if tokenizer.unk_id in ids:
        raise ValueError("input contains tokens outside receiver vocabulary")
    with torch.no_grad():
        v = model.encode_semantic(torch.tensor([ids], dtype=torch.long, device=chosen),
                                  pooling=pooling, hybrid_alpha=alpha,
                                  normalize_hybrid=normalized)[0]
    if v.ndim != 1 or v.numel() != 64 or not bool(torch.isfinite(v).all()):
        raise ValueError("invalid 64-dimensional receiver semantic vector")
    return {"vector": v.detach().cpu().tolist(), "dimension": 64,
            "token_count": len(ids), "device": chosen, "pooling": pooling,
            "encoder_sha256": digest(checkpoint), "tokenizer_sha256": digest(tokenizer_path)}

def process_received(message, *, receiver, checkpoint, tokenizer_path, device="auto"):
    local_sd = import_message(message, receiver_llm=receiver)
    text = validate(local_sd)
    representation = encode_japanese(text, checkpoint=checkpoint,
                                     tokenizer_path=tokenizer_path, device=device)
    # Keep the local vector and encoder fingerprint private to the receiver SD.
    local_sd["receiver_semantic"] = representation
    local_sd["semantic_mode"] = "real-llm-sem-encoder"
    answer = receipt(local_sd, result_text=text)
    answer["processing_mode"] = "real-llm-sem-encoder"
    validate_receipt(message, answer)
    return local_sd, answer

def run_messages(messages, models, *, device="auto"):
    if not isinstance(messages, list) or len(messages) < 2 or not isinstance(models, dict):
        raise ValueError("two or more messages and model mappings required")
    # Validate the complete batch before loading any expensive model weights.
    targets = []
    for message in messages:
        if not isinstance(message, dict):
            raise ValueError("each message must be a JSON object")
        target = message.get("target_llm")
        if not isinstance(target, str) or not target:
            raise ValueError("missing receiver ID")
        targets.append(target)
    if len(set(targets)) != len(targets):
        raise ValueError("duplicate receiver")
    for target in targets:
        spec = models.get(target)
        if not isinstance(spec, dict) or not spec.get("checkpoint") or not spec.get("tokenizer"):
            raise ValueError("checkpoint and tokenizer required for " + target)
    outputs = []
    for message, target in zip(messages, targets):
        spec = models[target]
        sd, response = process_received(message, receiver=target,
            checkpoint=spec["checkpoint"], tokenizer_path=spec["tokenizer"], device=device)
        outputs.append({"receiver": target, "local_sd": sd, "response": response,
                        "reply_verified": validate_receipt(message, response) == validate(sd)})
    return {"version": "v0.7", "mode": "real-llm-sem-encoder",
            "all_receipts_valid": all(row["reply_verified"] for row in outputs),
            "semantic_equivalence": "not-evaluated", "results": outputs}

def main():
    p = argparse.ArgumentParser(description="v0.7 real receiver-side LLM_SEM encoding")
    p.add_argument("messages", help="JSON list of SKI message objects")
    p.add_argument("models", help="JSON map from receiver LLM ID to checkpoint and tokenizer")
    p.add_argument("output")
    p.add_argument("--device", choices=["auto","cpu","cuda"], default="auto")
    a = p.parse_args()
    try:
        messages = json.loads(Path(a.messages).read_text(encoding="utf-8-sig"))
        models = json.loads(Path(a.models).read_text(encoding="utf-8-sig"))
        result = run_messages(messages, models, device=a.device)
        out = Path(a.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
        print(json.dumps({"mode":result["mode"], "all_receipts_valid":result["all_receipts_valid"],
              "semantic_equivalence":result["semantic_equivalence"],
              "results":[{"receiver":r["receiver"],"dimension":r["local_sd"]["receiver_semantic"]["dimension"],
                          "token_count":r["local_sd"]["receiver_semantic"]["token_count"]} for r in result["results"]]},
              ensure_ascii=False, indent=2))
    except (OSError, ValueError, TypeError, KeyError, RuntimeError) as e:
        p.exit(2, f"error: {e}\n")

if __name__ == "__main__":
    main()
