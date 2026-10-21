"""Train and run a conditional Japanese GRU decoder for LLM_SEM 64-D vectors.

A decoder is model-specific. It must be trained on vectors produced by the
same source checkpoint, tokenizer, pooling and normalization settings.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "vendor" / "LLM_SEM"))
from model import LanguageModel
from tokenizer import Tokenizer


class SemanticDecoder(nn.Module):
    def __init__(self, vocab_size: int, semantic_dim: int = 64, hidden_size: int = 256, layers: int = 2):
        super().__init__()
        self.semantic_dim = semantic_dim
        self.hidden_size = hidden_size
        self.layers = layers
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.context = nn.Linear(semantic_dim, hidden_size)
        self.initial = nn.Linear(semantic_dim, layers * hidden_size)
        self.gru = nn.GRU(hidden_size * 2, hidden_size, layers, batch_first=True)
        self.out = nn.Linear(hidden_size, vocab_size)

    def forward(self, vectors, tokens):
        if vectors.ndim != 2 or vectors.size(1) != self.semantic_dim:
            raise ValueError(f"expected vectors of shape [batch,{self.semantic_dim}]")
        context = torch.tanh(self.context(vectors)).unsqueeze(1).expand(-1, tokens.size(1), -1)
        h = torch.tanh(self.initial(vectors)).reshape(-1, self.layers, self.hidden_size).transpose(0, 1).contiguous()
        output, _ = self.gru(torch.cat([self.embedding(tokens), context], dim=-1), h)
        return self.out(output)

    @torch.no_grad()
    def generate(self, vector, tokenizer, max_tokens=96):
        self.eval()
        if max_tokens < 1:
            raise ValueError("max_tokens must be positive")
        if vector.ndim == 1:
            vector = vector.unsqueeze(0)
        if vector.shape != (1, self.semantic_dim) or not torch.isfinite(vector).all():
            raise ValueError(f"requires one finite {self.semantic_dim}-dimensional vector")
        ids = [tokenizer.bos_id]
        for _ in range(max_tokens):
            token_input = torch.tensor([ids], dtype=torch.long, device=vector.device)
            logits = self(vector, token_input)[0, -1].clone()
            logits[tokenizer.pad_id] = -float("inf")
            logits[tokenizer.bos_id] = -float("inf")
            next_id = int(logits.argmax().item())
            if next_id == tokenizer.eos_id:
                break
            ids.append(next_id)
        return tokenizer.decode(ids[1:])


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1048576), b""):
            h.update(chunk)
    return h.hexdigest()


def read_corpus(path):
    p = Path(path)
    if p.suffix.lower() == ".jsonl":
        lines = [json.loads(line)["text"] for line in p.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    else:
        lines = p.read_text(encoding="utf-8-sig").splitlines()
    result = list(dict.fromkeys(x.strip() for x in lines if isinstance(x, str) and x.strip()))
    if len(result) < 2:
        raise ValueError("corpus must contain at least two distinct nonempty sentences")
    return result


def load_encoder(checkpoint, tokenizer_path, device):
    tokenizer = Tokenizer.load(str(tokenizer_path))
    encoder, _ = LanguageModel.load_checkpoint(str(checkpoint), device)
    encoder.eval()
    if encoder.d_model != 64:
        raise ValueError(f"source model has dimension {encoder.d_model}; 64 required")
    if encoder.vocab_size != tokenizer.vocab_size:
        raise ValueError("source tokenizer/model vocabulary mismatch")
    return encoder, tokenizer


@torch.no_grad()
def encode_pairs(texts, encoder, tokenizer, pooling, alpha, normalized, max_length):
    vectors, sequences = [], []
    for text in texts:
        ids = tokenizer.encode(text, add_bos=True, add_eos=True)
        if len(ids) > min(max_length, encoder.context_length):
            continue
        if tokenizer.unk_id in ids:
            continue
        tokens = torch.tensor([ids], dtype=torch.long, device=next(encoder.parameters()).device)
        vector = encoder.encode_semantic(tokens, pooling=pooling, hybrid_alpha=alpha, normalize_hybrid=normalized)[0]
        vectors.append(vector.detach().cpu())
        sequences.append(ids)
    if not vectors:
        raise ValueError("no usable sentences: check tokenizer coverage and encoder context length")
    return torch.stack(vectors), sequences


def train(args):
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device))
    encoder, tokenizer = load_encoder(args.encoder, args.tokenizer, device)
    if args.max_length > encoder.context_length:
        raise ValueError("max-length cannot exceed source encoder context_length")
    texts = read_corpus(args.corpus)
    vectors, sequences = encode_pairs(texts, encoder, tokenizer, args.pooling, args.alpha, args.normalized, args.max_length)
    if len(sequences) < 2:
        raise ValueError("need at least two encodable sentences")
    decoder = SemanticDecoder(tokenizer.vocab_size, hidden_size=args.hidden, layers=args.layers).to(device)
    optimizer = torch.optim.AdamW(decoder.parameters(), lr=args.lr)
    order = torch.randperm(len(sequences), generator=torch.Generator().manual_seed(args.seed)).tolist()
    val_count = max(1, int(len(order) * args.val_ratio))
    val_idx, train_idx = order[:val_count], order[val_count:]
    if not train_idx:
        raise ValueError("empty training split")
    def epoch(indices, training):
        decoder.train(training)
        running, count = 0.0, 0
        for offset in range(0, len(indices), args.batch_size):
            selected = indices[offset:offset + args.batch_size]
            width = max(len(sequences[i]) for i in selected)
            batch = torch.full((len(selected), width), tokenizer.pad_id, dtype=torch.long, device=device)
            for k, idx in enumerate(selected):
                batch[k, :len(sequences[idx])] = torch.tensor(sequences[idx], device=device)
            vec = vectors[selected].to(device)
            with torch.set_grad_enabled(training):
                logits = decoder(vec, batch[:, :-1])
                loss_sum = F.cross_entropy(logits.reshape(-1, logits.size(-1)), batch[:, 1:].reshape(-1), ignore_index=tokenizer.pad_id, reduction="sum")
                n = int((batch[:, 1:] != tokenizer.pad_id).sum().item())
                loss = loss_sum / n
                if training:
                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(decoder.parameters(), 1.0)
                    optimizer.step()
            running += float(loss_sum.detach())
            count += n
        return running / max(count, 1)
    best = float("inf")
    for number in range(1, args.epochs + 1):
        tr = epoch(train_idx, True)
        with torch.no_grad():
            val = epoch(val_idx, False)
        print(f"epoch={number} train_nll={tr:.4f} val_nll={val:.4f}", flush=True)
        if val < best:
            best = val
            path = Path(args.output)
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                "format": "semos-s2n-decoder-v0.1",
                "model_state_dict": decoder.state_dict(),
                "config": {"semantic_dim": 64, "hidden_size": args.hidden, "layers": args.layers, "vocab_size": tokenizer.vocab_size},
                "source": {"encoder_sha256": digest(args.encoder), "tokenizer_sha256": digest(args.tokenizer), "pooling": args.pooling, "hybrid_alpha": args.alpha, "normalized": args.normalized},
                "validation_nll": best,
                "training_samples": len(train_idx),
                "validation_samples": len(val_idx),
            }, path)
    print(f"saved={args.output} best_val_nll={best:.4f}")


def decode(args):
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device))
    try:
        saved = torch.load(args.decoder, map_location=device, weights_only=True)
    except FileNotFoundError:
        raise ValueError("decoder checkpoint not found: first run the train command")
    if saved.get("format") != "semos-s2n-decoder-v0.1":
        raise ValueError("unknown decoder checkpoint format")
    meta = saved["source"]
    if digest(args.encoder) != meta["encoder_sha256"] or digest(args.tokenizer) != meta["tokenizer_sha256"]:
        raise ValueError("encoder/tokenizer mismatch with trained decoder")
    tokenizer = Tokenizer.load(args.tokenizer)
    config = saved["config"]
    decoder = SemanticDecoder(config["vocab_size"], config["semantic_dim"], config["hidden_size"], config["layers"]).to(device)
    decoder.load_state_dict(saved["model_state_dict"])
    item = json.loads(Path(args.input).read_text(encoding="utf-8-sig"))
    if not isinstance(item, dict) or not isinstance(item.get("vector"), list):
        raise ValueError("input must contain a numeric vector")
    if item.get("dimension", 64) != 64:
        raise ValueError("semantic dimension mismatch")
    for field, expected in [("pooling", meta["pooling"]), ("normalized", meta["normalized"]), ("hybrid_alpha", meta["hybrid_alpha"])]:
        if field in item and item[field] != expected:
            raise ValueError(f"SemanticData {field} differs from decoder training")
    vector = torch.tensor(item["vector"], dtype=torch.float32, device=device)
    result = decoder.generate(vector, tokenizer, args.max_tokens)
    print(result)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Train/infer a 64-D LLM_SEM to Japanese decoder")
    subs = parser.add_subparsers(dest="action", required=True)
    t = subs.add_parser("train")
    t.add_argument("--corpus", required=True)
    t.add_argument("--epochs", type=int, default=30)
    t.add_argument("--batch-size", type=int, default=16)
    t.add_argument("--lr", type=float, default=0.001)
    t.add_argument("--hidden", type=int, default=256)
    t.add_argument("--layers", type=int, default=2)
    t.add_argument("--max-length", type=int, default=64)
    t.add_argument("--val-ratio", type=float, default=0.1)
    t.add_argument("--seed", type=int, default=42)
    t.add_argument("--pooling", choices=["hybrid", "mean", "last", "bos", "max", "attention"], default="hybrid")
    t.add_argument("--alpha", type=float, default=0.35)
    t.add_argument("--normalized", action="store_true")
    t.add_argument("--output", default="model/semantic-decoder-v01.pt")
    d = subs.add_parser("decode")
    d.add_argument("input")
    d.add_argument("--decoder", default="model/semantic-decoder-v01.pt")
    d.add_argument("--max-tokens", type=int, default=64)
    for sub in (t, d):
        sub.add_argument("--encoder", default="vendor/LLM_SEM/model/model-gpu-v0.4.pt")
        sub.add_argument("--tokenizer", default="vendor/LLM_SEM/model/tokenizer.json")
        sub.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = parser.parse_args(argv)
    try:
        if args.action == "train":
            if not 0 < args.val_ratio < 1 or args.epochs < 1 or args.batch_size < 1:
                raise ValueError("invalid training configuration")
            train(args)
        else:
            decode(args)
    except (ValueError, FileNotFoundError, KeyError) as exc:
        parser.exit(2, f"error: {exc}\n")


if __name__ == "__main__":
    main()
