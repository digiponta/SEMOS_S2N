# SEMOS_S2N v0.9 — Model-local Semantic Memory Baseline

Receivers register Japanese propositions as **local 64D vectors** with their
encoder + tokenizer fingerprints. Known/Unknown decisions use cosine
similarity **only within the same receiver model**, never across models.

- `LocalMemory.insert(text, encoded)`: idempotent registration, fingerprint guard.
- `LocalMemory.query(text, encoded)`: nearest item with threshold or UNKNOWN.
- `evaluate(examples, models)`: run a teach/query sequence and labeled accuracy.
- No on-disk persistent memory yet; evaluation memory lives in process.
- Unknown gating is a *baseline*, not calibrated: a high similarity can still
  yield a false acceptance. Exact-text matches are the easy control cases.
- No proof of semantic understanding, paraphrase equivalence, or HyperAether
  network execution.

```powershell
git pull --ff-only origin main
python -m unittest -v test_semos_memory_v09.py
python semos_memory_v09.py data/memory_v09_examples.json receiver_models.json results/memory_v09.json --device cuda --threshold 0.85
```

The example leaves open-ended unknown queries unlabeled deliberately.
Evaluate with an independently labeled known/paraphrase/unknown benchmark
before making accuracy claims. This integrates encoding and in-memory
retrieval only; the receiver does not yet perform autonomous inference.
