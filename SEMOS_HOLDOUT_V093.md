# v0.9.3 — Frozen distinct-data evaluation

The existing v0.9.2 rule gate is left unchanged. Twelve new authored
examples probe paraphrases, negation variants, unseen subjects, out-of-domain
texts and lexical overlap. This is a **separate authored set**, not a true
independently sampled corpus. Do not adjust rules and then still describe it
as untouched holdout.

Metrics:
- **coverage** = decided (KNOWN/UNKNOWN) / all
- **selective_accuracy** = correct decisions / decided, excludes REVIEW
- **strict_accuracy** = correct decisions / all, counts REVIEW as noncorrect
- **known_recall**, **unknown_reject_rate** count REVIEW as noncorrect
- **unknown_false_accept_rate** = falsely KNOWN / all labeled UNKNOWN

RUN
```powershell
git pull --ff-only origin main
python -m unittest -v test_semos_holdout_v093.py
python semos_holdout_v093.py data/holdout_v093.json receiver_models.json results/holdout_v093.json --device cuda --threshold 0.85
```

Expected: out-of-domain paraphrases and lexical variants may abstain; unseen
subject (TPU/NPU) will likely result in REVIEW because v0.9.2 has a
deliberately restricted parser. This is a robustness diagnostic rather than
an attempt to maximize the small test score.
