# SEMOS_S2N — Scope and Exchange Contract (2026-10-21)

## Objective

SEMOS_S2N's responsibility is to convey **Dataset A**, a collection of natural-language propositions plus associated attributes, to a recipient LLM in a verifiable, parseable format. The recipient's interpretation, reasoning, learning, and semantic agreement are **out of scope**. Its response **Dataset B must use the same dataset schema** as A. A and B may contain different propositions and attributes.

**A valid transport/response exchange is not evidence of semantic equivalence or truth.**

## Dataset schema proposal: s2n-dataset-1

```json
{
  "schema_version": "s2n-dataset-1",
  "dataset_id": "ds-A-0001",
  "sender_id": "LLM-A",
  "recipient_id": "LLM-B",
  "in_reply_to": null,
  "propositions": [
    {
      "proposition_id": "p-A-1",
      "text": "GPUは並列計算に適している。",
      "attributes": {
        "language": "ja",
        "source_sd_id": "sd-example",
        "provenance": "LLM-A"
      }
    }
  ]
}
```

A response B uses the **same** `schema_version`, structure, and property types,
with a new `dataset_id`, a `sender_id` representing the responding model,
and `in_reply_to` identifying Dataset A. No requirement for equal text or
equal proposition counts. Attributes are JSON-serializable metadata and are
preserved as transmitted, with no implied truth or semantic interpretation.

## Sender guarantees

1. Validate A against a shared versioned schema before encoding/transmission.
2. Protect payload integrity (e.g., deterministic serialization plus hash).
3. Verify receipt of the complete A bytes or an equivalent canonicalized
   representation; a delivery acknowledgement must refer to A's ID/digest.
4. Correlate B to A using `in_reply_to`, validate B against the same schema
   and its integrity metadata.
5. Reject malformed, truncated, altered, wrong-recipient or mismatched-reply
   envelopes; report failures as protocol/transport errors, not semantic errors.

A receipt acknowledgment establishes **protocol-level receipt** only.
It does not prove the model processed, understood, or accepted a proposition.
When exchange occurs over a local simulator, network delivery guarantees have
not yet been tested.

## Recipient's freedom and minimal obligation

- Free to interpret, ignore, transform, infer from, or decline to use A.
- If it responds through S2N, its response B must conform to the shared
  dataset schema; no mandated semantic relation to A.
- A receiver adapter may wrap a model-native reply into Dataset B. The
  model's internal format is not prescribed.

## Explicitly out of scope

- Cosine similarity of cross-model vectors.
- Semantic equivalence of the original and transformed propositions.
- Known/Unknown and Truth-Aware Gate accuracy.
- Encoder checkpoint compatibility, common vector basis, model training.

Existing v0.7–v0.9.3 model-local tests remain optional experiments and
should not serve as acceptance criteria for S2N interchange.

## Next implementation acceptance tests

- A round trip: valid A -> recipient acknowledgment -> schema-valid B.
- Same structure, different A/B propositions; zero or multiple propositions.
- Custom nested attributes preserved.
- Missing text/IDs, malformed attribute JSON, unknown schema version rejected.
- Tampered digest, duplicate reply ID, mismatched `in_reply_to`, wrong
  recipient rejected.
- Timeout and failure conditions represented as explicit protocol errors.
