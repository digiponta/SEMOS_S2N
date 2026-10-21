# SEMOS_S2N v0.9.2 — Restricted Truth-Aware Proposition Gate

A conservative, **rule-based prototype** combines receiver-local cosine with
explicit Japanese proposition subject, relation and polarity checks.

Outcomes: KNOWN (structure and cosine agree), UNKNOWN (subject, relation or
polarity differs), REVIEW (parser unsupported, ambiguous or low similarity).
REVIEW is abstention, not validated unknown knowledge.

The initial parser recognizes only limited GPU/CPU parallel-computing and
machine-learning-utility examples. It cannot parse arbitrary Japanese,
resolve scope reliably, or establish real-world truth. A changed subject is
treated as not the registered proposition, not automatically a false claim.

Commands:
```powershell
git pull --ff-only origin main
python -m unittest -v test_semos_truth_gate_v092.py
python semos_truth_gate_v092.py data/semantic_benchmark_v091.json receiver_models.json results/truth_gate_v092.json --device cuda --threshold 0.85
```

The benchmark is the **same ten cases used to motivate these rules**, so its
scores must be marked **in-sample only** and not generalization. Next: design
a structurally diverse frozen holdout set before modifying rules, measure
coverage and selective accuracy, and investigate dedicated NLI/slot parsing.
