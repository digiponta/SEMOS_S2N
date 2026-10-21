# v0.8 Next Experiment — Distinct checkpoint smoke test

The existing experiment has **identical SHA-256 checkpoints** for LLM-B and
LLM-C. To exercise the heterogeneous *checkpoint identity* path without
inventing a second real pretrained model, create a controlled variant of the
same LLM_SEM model. This changes one floating model parameter tensor by
very small seeded Gaussian noise, retains architecture/tokenizer compatibility,
and **does not** constitute independently trained heterogeneous LLMs.

```powershell
git pull --ff-only origin main
python prepare_receiver_variant_v08.py
```

Edit `receiver_models.json` so ONLY LLM-C uses:
`vendor/LLM_SEM/model/model-gpu-v0.4-variant.pt`

Then run:

```powershell
python semos_heterogeneous_v08.py ski_messages_v07.json receiver_models.json results/heterogeneous_v08_variant.json --device cuda
```

Expected `distinct_model_fingerprints=true`; this checks differing model
identities, exact natural-language transport and local vector generation.
Semantic equivalence remains `not-evaluated`. For a proper heterogeneous
LLM study, use independently trained compatible models or a separate
model-specific adapter; do not present this perturbation as such.

**Security note:** Load checkpoints only from trusted origins. PyTorch
checkpoint deserialization can execute code. Output weights are intentionally
local only; do not commit binary checkpoints to Git.
