# Semantic Decoder v0.1 — 64-D LLM_SEM → Japanese

A **trainable**, conditional GRU decoder. It does **not** reverse embeddings by mathematics alone: it needs paired training data `(LLM_SEM vector, original Japanese sentence)`. It is separate from the deterministic renderer `semantic_to_japanese.py`.

## Preparation
```powershell
cd C:\Users\inoma\git\misc\SEMOS_S2N
git pull origin main
python -m pip install -r vendor/LLM_SEM/requirements.txt
```

Copy the matching LLM_SEM checkpoint and tokenizer (not committed with source) into:
- `vendor/LLM_SEM/model/model-gpu-v0.4.pt`
- `vendor/LLM_SEM/model/tokenizer.json`

**Important:** Use the exact model/tokenizer pair used to create the target vectors. Do not use arbitrary checkpoints or tokenizer vocabularies. Source model dimensionality must equal **64**. If the source checkpoint context length is 64, sentence tokens including BOS/EOS must fit.

Prepare a UTF-8 training corpus: one **different Japanese sentence per line** in `data/decoder_train.txt`, or a JSONL file with `{"text":"..."}` on each line. Aim for a sizeable, diverse corpus. For an initial smoke test:
```powershell
New-Item -ItemType Directory -Force data | Out-Null
@'
猫は動物である。
GPUは高速な計算装置である。
量子センサーは微小な物理量を測定する。
Semantic OSは意味情報を使って処理を制御する。
'@ | Set-Content -Encoding utf8 data/decoder_train.txt
```

### Training

```powershell
python semantic_decoder.py train --corpus data/decoder_train.txt --epochs 30 --device auto
```

The program constructs source semantic vectors using `LanguageModel.encode_semantic` with the same default `pooling=hybrid`, `hybrid_alpha=0.35` and `normalized=False` used in `LLM_SEM/semantic.py`. It excludes examples longer than the source context and sentences with unknown tokens, does a seeded sentence-level holdout, trains with teacher forcing and cross-entropy, and saves the decoder with the best validation NLL.

### Decode an actual vector

Input file `semantic_vector.json`:
```json
{"vector":[/* 64 real numbers here */],"dimension":64,"pooling":"hybrid","hybrid_alpha":0.35,"normalized":false}
```

```powershell
python semantic_decoder.py decode semantic_vector.json --max-tokens 64
```

The checkpoint SHA-256 fingerprints of the encoder and tokenizer must match the training source. Inference requires a **trained** `model/semantic-decoder-v01.pt`. A decoder trained on a tiny corpus may memorize sentences rather than generalize. Evaluate on held-out examples and compare exact match, character error rate and semantic preservation before using this decoder in SKI/HyperAether.

### Tests
```powershell
python -m unittest -v test_semantic_decoder.py
```

**Known limits:** This is an experimental decoder, not an exact inversion of a lossy 64-D representation. Different sentences can yield similar vectors. Hallucinated or truncated output is possible. Source vectors must be produced with the same encoder, tokenizer and pooling policy used at training.
