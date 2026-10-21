# SEMOS_S2N v0.3 — Deep RSD reconstruction and size study

New module: `rsd_v03.py`. The `rsd.py` v0.2 schema and existing CLI remain compatible.

- `deep_split(sd, depth)`: recursively bisects SD with parent-link metadata at each level.
- `reconstruct_leaves(parts)`: bottom-up merge from unordered leaves; detects missing, duplicated and corrupt parts.
- `pack(sd)` / `unpack(envelope)`: optional reversible UTF-8 + zlib + base64 transport envelope. Enforces an uncompressed-byte limit and validates the original SHA-256.
- `benchmark(text, depth)`: exact reconstruction and byte-size measures for source text, JSON SD, compressed envelope, and distributed leaf records.

Run from PowerShell:

```powershell
git pull origin main
python -m unittest -v test_rsd.py test_rsd_v03.py test_semantic_to_japanese.py
python rsd_v03.py original.txt --depth 4 --output results/rsd_v03_benchmark.json
```

**Interpretation:** Compression must be assessed using the actual `zlib_envelope_bytes`, including base64 and JSON overhead. Smaller input can expand. `all_leaf_json_bytes` is a sum across leaf SD records, including duplicated reconstruction metadata. The test is a *storage-format* and *reconstruction* benchmark; it does not claim that canonical semantic meaning alone can reproduce an arbitrary original Japanese string. Source text bytes, rather than text after BOM stripping, require a separate byte-preserving format if absolute file-byte round trips are needed.
