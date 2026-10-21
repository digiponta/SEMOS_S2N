# SEMOS_S2N v0.6 — Distributed Processing Adapter Boundary

This release adds a transport-independent multi-recipient dispatcher and
callback interface for model-specific processing.

- Source LLM-A generates proposition SD(B) and SD(C) by the existing v0.4 route.
- Each SD is rendered independently to Japanese X and exported via a v0.5
  SKI message. The v0.4 `source_witness` is omitted from transport.
- Per-target import creates a receiver-local SD with provenance.
- A registered adapter callback receives Japanese X and receiver ID. It can
  call an external model and return Japanese or a valid receiver-owned SD.
- Replies are correlated and their text hashes verified before aggregation.

**Status:** By default the adapter is an **echo simulator**. No inference,
networking, parallel execution, authentication, or semantic equivalence
verification is implemented. Neither SHA-256 nor deterministic IDs are
authentication. Omission of the source witness prevents remote receivers
from independently reconstructing the original A; only individual X is
reversible on the receiving side.

## PowerShell

```powershell
git pull origin main
python -m unittest -v test_semos_distributed_v06.py test_ski_nl_exchange.py test_semantic_proposition_sd.py test_rsd.py test_rsd_v03.py test_semantic_to_japanese.py
python semos_distributed_v06.py derived_sd.json results/distributed_v06.json --receivers LLM-B LLM-C
```

An actual model integration can pass a callable `adapter(text, receiver_id)`
through `run(parts, receivers, adapters={"LLM-B": adapter_b, "LLM-C": adapter_c})`.
The callable is responsible for real re-semantification/inference and
model-specific SD generation.

Next: LLM_SEM adapter, real inference measurement, semantic preservation
tests, authorization and verified network transport, parallel scheduling,
result fusion, and minimized auxiliary reconstruction data.
