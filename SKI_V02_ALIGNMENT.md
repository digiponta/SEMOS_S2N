# SEMOS_S2N — SKI v0.2 draft alignment (2026-10-21)

## Sources and versions verified

- **HA_DOC main**, latest commit `8eae29f9b6fe7f1f1c8d063de7a676e3cf1d812e`:
  `HyperAether_Revised_Distributed_LLM_Architecture_20261020.docx`.
  Contains revised HyperAether architecture work. No SKI schema file in the
  commits inspected.
- **HASKI_DOC main**, latest commit
  `501a0da84ada01617c2dee0980bb63b6d81f6b68`:
  `docs/SKI_v0.2_Core_Data_Model_Lifecycle_Distributed_Synchronization_ja.md`
  is the latest verified **SKI specification document** (2026-10-19).

Source:
https://github.com/digiponta/HASKI_DOC/blob/main/docs/SKI_v0.2_Core_Data_Model_Lifecycle_Distributed_Synchronization_ja.md

## Required SKI v0.2 core attributes

Each KnowledgeAssertion uses JSON-LD fields `@id`, `@type`,
`ski:claim` (subject / predicate / object), `ski:issuer`,
`ski:recordedAt` (timezone-aware), `ski:version` (>=1), and
`ski:lifecycle`. Other attributes, e.g., `ski:conditions`,
`ski:context`, `prov:wasDerivedFrom`, and `ski:assessments`,
are passed through unchanged.

The specification distinguishes assertion **receipt** from **approval**,
**assessment**, **synchronization commit**, and **model internalization**.
A and B may differ in meaning, and no S2N semantic check is required.

## Prototype alignment and restrictions

`ski_core_v02_exchange.py` provides a **local partial structural validator**
and an in-process A/ACK/B demonstration. It is **not official SKI v0.2
conformance**: machine-readable canonical JSON-LD context, official SKI IRI,
SHACL shape, signatures, authentication, full lifecycle/sync rules and remote
transfer remain unpublished/unimplemented.

- The source SKI v0.2 JSON-LD example uses `example.org` placeholders.
- `s2n:naturalLanguage` is a **clearly identified S2N extension**, not an
  officially defined SKI vocabulary term.
- `s2n:inReplyTo` is a **clearly identified S2N correlation extension**, not
  an official SKI v0.2 term. The formal integration must define extension
  policy with SKI authors before calling it interoperable.
- Dataset A and B both use **lists of SKI v0.2 KnowledgeAssertion JSON-LD**,
  not the legacy `ski-natural-language-v0.5` transport object.
- The acknowledgment structure is local protocol metadata, **not** a
  standardized SKI-Sync COMMITTED event.
- Legacy `ski_dataset_roundtrip.py` is retained for backwards test results
  but is **not** presented as SKI v0.2 standard compliant.

## Local test

```powershell
git pull --ff-only origin main
python -m unittest -v test_ski_core_v02_exchange.py
```

The tests validate structural acceptance/rejection, attribute retention,
ACK integrity and request-response correlation. No actual remote LLM is
executed.
