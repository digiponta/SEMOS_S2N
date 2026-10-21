# SKI v0.5 Dataset A/B Roundtrip — Local Prototype

Both Dataset A and Dataset B are **JSON arrays of existing SKI v0.5
`ski-natural-language-v0.5` messages** (no novel S2N wire schema).
Transport acknowledgment is **separate** from Dataset B. The example
recipient adapter returns caller-provided texts: it does not invoke an LLM
or check interpretation.

The current checked-in `ski_nl_exchange.py` is a prototype **per-message**
format. It has not been established that an array of its messages conforms
to an authoritative HASKI_DOC dataset schema. Do not claim full SKI v0.4/v0.5
dataset standard conformance until the canonical specification is integrated.

```powershell
git pull --ff-only origin main
python -m unittest -v test_ski_dataset_roundtrip.py
python prepare_ski_roundtrip_examples.py
python ski_dataset_roundtrip.py data/ski_dataset_a_example.json data/ski_replies_example.json results/ski_roundtrip_local.json --receiver LLM-B
```

Verifies per-message IDs/hashes, transport ACK digest, reply routing and
correlation using the existing `derivation` attributes, separate reply
content and the same SKI v0.5 message layout on both sides.

Limitations: This is an **in-process exchange simulation**, not verified
remote delivery; reply adapter currently has a one-to-one sample constraint;
the result file contains ACK and Dataset B for display but is not itself a
transmitted SKI dataset envelope. Production implementation must pin the
HASKI_DOC-defined SKI dataset container, support arbitrary proposition counts,
and implement a real receiver transport.
