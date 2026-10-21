# SEMOS_S2N: Semantic Data → 日本語

Source: [LLM_SEM v0.20.0](https://github.com/digiponta/LLM_SEM/tree/v0.20.0), SHA `8f5a2822c35296274813660da65df2176072ec94`.

Convert JSON SemanticData, including `text`, or symbolic propositions with `subject` and `value`. Supported relations: `is`, `includes`, `has`.

```powershell
python semantic_to_japanese.py semantic.json
python semantic_to_japanese.py semantic.jsonl --jsonl -o output-ja.txt
python -m unittest -v test_semantic_to_japanese.py
```

Example input: `{"subject":"量子センサー","value":"高感度"}`

Output: `量子センサーは、高感度である。`

**Note:** Vector-only data is rejected because LLM_SEM embedding coordinates cannot be decoded reliably without a compatible learned decoder. A saved source text or symbolic proposition is required.
