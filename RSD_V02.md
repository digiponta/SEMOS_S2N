# SEMOS_S2N v0.2 — Reconstructable Semantic Data

RSD treats **exact Japanese text reconstruction** and **semantic interpretation** separately.

An SD record retains a lossless payload, UTF-8 SHA-256 digest and identifier. When split, **both** child SDs carry the parent identity, parent checksum, part count, ordering, join operation and original length. Merge rejects mismatched, damaged, duplicated or missing fragments. Split is by Unicode code-point index; the payload preserves exact spaces and punctuation. This is a *lossless baseline*, not a compression result.

In this version `A = Merge(B, C)` means structured concatenation rather than addition of vector coordinates. A child is independently verifiable, and a previously reconstructed child may be merged at its higher level, supporting hierarchical splitting.

## PowerShell example

```powershell
git pull origin main
"GPUは並列計算に適している。" | Set-Content -Encoding utf8 original.txt
python rsd.py encode original.txt sd_a.json
python rsd.py split sd_a.json 5 sd_b.json sd_c.json
python rsd.py merge sd_c.json sd_b.json sd_restored.json
python rsd.py decode sd_restored.json restored.txt
python -m unittest -v test_rsd.py test_semantic_to_japanese.py test_semantic_decoder.py
```

The `restored.txt` file contains the original Unicode text contents (BOM on the source file is treated as an encoding marker, not preserved). Note that PowerShell `Set-Content` typically adds a terminal newline; this is also preserved.

**Scope:** Current `rsd.py` supports binary splits and validated hierarchical merges. It does not automatically extract canonical propositions, compress SD, preserve source file's original byte encoding/BOM, or prove semantic equivalence. For round-trip byte identity, add a raw-byte representation in a future version. The existing GRU Decoder remains an experimental vector-only alternative.
