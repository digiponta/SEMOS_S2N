# SEMOS_S2N v0.9.1 — Labeled Known/Unknown Evaluation

This version evaluates ten Japanese queries across LLM-B and LLM-C using
five test categories: exact, paraphrase, contradiction, subject change, and
unrelated. Each receiver registers exactly one known sentence and classifies
queries with its **own** checkpoint's cosine gate.

Run:

```powershell
git pull --ff-only origin main
python -m unittest -v test_semos_benchmark_v091.py
python semos_benchmark_v091.py data/semantic_benchmark_v091.json receiver_models.json results/semantic_benchmark_v091.json --device cuda --threshold 0.85
```

The benchmark reports known recall, unknown rejection rate, balanced accuracy,
per-category accuracy and individual similarities. These are **small,
illustrative human-assigned labels**, not an independently validated final
benchmark. Negation and subject changes can be falsely accepted at high cosine.
Do not tune a threshold on the same ten examples and describe the resulting
numbers as held-out generalization. A later version should add disjoint
calibration/evaluation sets, stronger semantic contradiction detection,
independently trained models, and multiple propositions per receiver.
