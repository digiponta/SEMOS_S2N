# SEMOS_S2N v0.5 — SKI Natural Language Exchange Prototype

Goal: Transform a source model's SD(X) into Japanese X, transfer a JSON message to another LLM environment, and recreate a receiver-local SD(X). Vectors are never exchanged. Sender/receiver model IDs, parent provenance, source SD ID and a correlation ID travel alongside the natural language.

New files: `ski_nl_exchange.py`, `test_ski_nl_exchange.py`.

The prototype **does not yet call an external LLM or implement network transport**. The receiver's `make_sd(text)` is a deterministic placeholder for model-specific semantic encoding. The result receipt is likewise a protocol example, not proof of distributed inference. Integrity hashes are error-detection checks, not cryptographic authentication. The copied `derivation` may include v0.4's full original sentence, so the transport is not compact.

PowerShell:

```powershell
git pull origin main
python -m unittest -v test_ski_nl_exchange.py test_semantic_proposition_sd.py test_rsd.py test_rsd_v03.py test_semantic_to_japanese.py
python -X utf8 -c "import json; a=json.load(open('derived_sd.json',encoding='utf-8-sig')); json.dump(a[0],open('sd_b.json','w',encoding='utf-8'),ensure_ascii=False,indent=2)"
python ski_nl_exchange.py export sd_b.json ski_b.json --source LLM-A --target LLM-B
python ski_nl_exchange.py import ski_b.json received_b.json --receiver LLM-B
python ski_nl_exchange.py receipt received_b.json receipt_b.json
```

Next phases: real model-specific adapters, authenticated SKI transport, cross-model semantic equivalence tests, distributed execution and results aggregation through HyperAether.
