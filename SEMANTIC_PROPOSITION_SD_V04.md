# SEMOS_S2N v0.4: AI-derived proposition SD

**Definition:** SD(A) denotes semantic data associated with Japanese A. AI transforms A into *new propositions* B and C—not string slices. SD(B) and SD(C) each independently reconstruct B and C. Their assembly reconstructs A using auxiliary lossless witness data.

This is a conservative reference implementation of the data contract: proposition discovery is supplied as input, not automatically executed by an AI model. Because B and C can paraphrase, expand or omit phrasing, A cannot generally be reconstructed exactly from B+C alone. Each SD carries `source_witness` for exact reconstruction plus parent hash, position, sibling identities, model identifier and derivation relation. This is **deliberately redundant**, prioritizing self-contained transfer. A compact shared-witness/residual variant is future work. Semantic equivalence of B/C to A is **not validated**.

Example `data/proposition_example.json`:
```json
{
  "source": "GPUは並列計算に適しているので、機械学習に有用である。",
  "propositions": [
    "GPUは並列計算に適している。",
    "GPUは機械学習に有用である。"
  ],
  "relation": "explanation",
  "model_id": "manual-demo"
}
```

PowerShell:
```powershell
git pull origin main
python -m unittest -v test_semantic_proposition_sd.py test_rsd.py test_rsd_v03.py test_semantic_to_japanese.py
python semantic_proposition_sd.py derive data/proposition_example.json derived_sd.json
python semantic_proposition_sd.py assemble derived_sd.json recovered_a.json
python semantic_to_japanese.py recovered_a.json
```

To decode SD(B) or SD(C) independently, extract one element from the JSON array into a JSON file, then run `python semantic_proposition_sd.py decode selected_sd.json`. Existing legacy `rsd.py` literal text splitting remains available but is a distinct operation.

**Research priorities:** generate candidate propositions with LLM, verify entailment/equivalence, optimize lossless witness/residual information and support multiple derivation links in SKI.
