"""Export a real 64-D SemanticData vector using the matching LLM_SEM encoder."""
import argparse
import json
import torch
from semantic_decoder import load_encoder, digest

def main():
    p = argparse.ArgumentParser()
    p.add_argument("text", help="Japanese input text")
    p.add_argument("-o", "--output", default="semantic_vector.json")
    p.add_argument("--encoder", default="vendor/LLM_SEM/model/model-gpu-v0.4.pt")
    p.add_argument("--tokenizer", default="vendor/LLM_SEM/model/tokenizer.json")
    p.add_argument("--pooling", default="hybrid")
    p.add_argument("--alpha", type=float, default=0.35)
    p.add_argument("--normalized", action="store_true")
    p.add_argument("--device", choices=["cpu", "cuda", "auto"], default="auto")
    args = p.parse_args()
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device))
    encoder, tokenizer = load_encoder(args.encoder, args.tokenizer, device)
    ids = tokenizer.encode(args.text, add_bos=True, add_eos=True)
    if len(ids) > encoder.context_length:
        p.error(f"token sequence {len(ids)} exceeds source context {encoder.context_length}")
    if tokenizer.unk_id in ids:
        p.error("input has tokens outside the source tokenizer vocabulary")
    with torch.no_grad():
        vector = encoder.encode_semantic(torch.tensor([ids], dtype=torch.long, device=device), pooling=args.pooling, hybrid_alpha=args.alpha, normalize_hybrid=args.normalized)[0]
    obj = {"vector": vector.detach().cpu().tolist(), "dimension":64, "token_count":len(ids), "pooling":args.pooling, "hybrid_alpha":args.alpha, "normalized":args.normalized,
        "encoder_sha256":digest(args.encoder), "tokenizer_sha256":digest(args.tokenizer)}
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    print(f"wrote {args.output} (dimension=64)")
if __name__ == "__main__":
    main()
