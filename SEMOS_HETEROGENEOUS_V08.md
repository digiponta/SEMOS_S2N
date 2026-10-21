# SEMOS_S2N v0.8 — Heterogeneous Model Fingerprint Audit

This experiment builds on the **real** LLM_SEM encoder adapter in v0.7.
For each receiver, it checks:
- its own checkpoint/tokenizer SHA-256 identity;
- that the input Japanese X survives local receipt;
- that the reply is correctly correlated and text-identical;
- 64D output dimension, token count and vector norm.

**Do not compare raw vectors from unlike models as if their coordinate
spaces were aligned.** This test does **not** establish semantic equivalence,
understanding, new inference, or distributed network operation. Its strength
is preventing a single-checkpoint smoke test from being mislabeled as a
heterogeneous-model experiment.

Commands:

```powershell
git pull origin main
python -m unittest -v test_semos_heterogeneous_v08.py test_semos_real_adapter_v07.py test_semos_distributed_v06.py test_ski_nl_exchange.py test_semantic_proposition_sd.py test_rsd.py test_rsd_v03.py test_semantic_to_japanese.py
python semos_heterogeneous_v08.py ski_messages_v07.json receiver_models.json results/heterogeneous_v08_smoke.json --device cuda --allow-identical
```

For a real cross-checkpoint study, edit `receiver_models.json` so LLM-C
points to another **compatible 64D** checkpoint and matching tokenizer,
then omit `--allow-identical`:

```powershell
python semos_heterogeneous_v08.py ski_messages_v07.json receiver_models.json results/heterogeneous_v08.json --device cuda
```

An unknown alternative checkpoint should not be invented or downloaded
without validating its architecture, tokenizer and trusted provenance.
Next phase: receiver-local retrieval tests on labeled known/unknown examples,
natural-language entailment verification, and real process isolation.
