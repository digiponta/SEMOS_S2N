# SEMOS_S2N — SKI Dataset Exchange Contract (2026-10-21)

## Scope and authoritative format

**Dataset A and response Dataset B MUST use the SKI data format.**
SEMOS_S2N does **not** introduce an independent `s2n-dataset-1` schema.
The earlier `s2n-dataset-1` example is **withdrawn**.

A dataset comprises natural-language proposition(s) and their associated SKI
attributes. Existing SKI definitions, including the relevant HASKI_DOC SKI
specifications, are authoritative for field names, identity, provenance and
dataset structure. Do not silently replace or rename SKI fields.

Sender A -> SKI dataset A -> receiver LLM -> SKI dataset B -> sender A.

## Guaranteed boundaries

1. Serialize and validate Dataset A under the **selected SKI version**.
2. Preserve the SKI natural-language proposition, attributes, identity and
   provenance as specified; detect corruption and routing mismatch.
3. Confirm receipt of the SKI dataset/message at the **protocol level**.
4. Accept Dataset B only if it satisfies **the same selected SKI dataset
   schema and validation rules as A**, allowing different proposition contents.
5. Correlate Dataset B with the request through the SKI-defined correlation
   and provenance mechanism.
6. The receiving LLM's interpretation, reasoning, memory and output meaning
   are entirely out of scope. A valid B need not be semantically equivalent
   to A.

No common embedding space, shared architecture or model compatibility is
required. The exchange layer transports natural-language SKI data, not
cross-model semantic vectors.

## Existing repository implementation versus target

The checked-in `ski_nl_exchange.py` v0.5 currently defines a **single
SKI message** tagged `ski-natural-language-v0.5` with
`message_id`, `source_llm`, `target_llm`, `source_sd_id`,
`parent_sd_id`, `language`, `text`, `text_sha256`, and `derivation`.
Its current `receipt()` creates a distinct
`ski-natural-language-v0.5-result` envelope containing `text`, NOT a
SKI dataset B of the same format as dataset A. Therefore the current code
**does not yet satisfy** this new A/B dataset exchange contract.

The previously proposed `schema_version`, `dataset_id`, `propositions`,
`attributes` wrapper was an S2N invention, **not a verified SKI format**.
Do not assume it is compatible with SKI. A new dataset container, if needed,
must be defined within an agreed SKI version, not as an S2N-specific wire
format.

## Required implementation changes

- Identify and pin the authoritative SKI dataset version/schema from
  HASKI_DOC before implementing or asserting compliance.
- Implement Dataset A serialization/validation using the SKI schema.
- Implement Dataset B serialization/validation using the **same SKI schema**.
- Separate transport-level receipt ACK from application-level Dataset B.
- Enforce SKI provenance/correlation and payload integrity.
- Test A -> ACK -> B, including arbitrary proposition/attribute contents,
  mismatched routing/correlation, corruption and invalid schemas.
- Preserve earlier v0.5 and v0.7–v0.9.3 tests as legacy experiments, but
  semantic-gate accuracy is **not** an exchange acceptance criterion.

## Definition of success

**The other LLM can receive an intact SKI dataset A and return a schema-valid
SKI dataset B.** What it inferred or understood from A is not tested by
SEMOS_S2N.
