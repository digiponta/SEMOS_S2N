# SEMOS_S2N v0.7 — Real LLM_SEM Receiver Encoder

`semos_real_adapter_v07.py` integrates the imported `vendor/LLM_SEM/model.py`
through the existing `semantic_decoder.load_encoder` function. The receiver
loads an actual local PyTorch checkpoint and tokenizer, tokenizes Japanese X,
and computes the **receiver-local 64-dimensional semantic vector** through
`LanguageModel.encode_semantic` with `torch.no_grad()`.

Unlike v0.6's echo-only processing, this path really runs the LLM_SEM encoder
**when checkpoint and tokenizer files are installed locally**. It does not
perform generative inference, validate semantic equivalence, update Semantic
Memory or invoke remote models. The Japanese reply is an acknowledgement
carrying the unchanged text, not a generated answer. Test mocks validate the
adapter boundary without needing GPU weights.

## Setup / verify

```powershell
git pull origin main
python -m unittest -v test_semos_real_adapter_v07.py test_semos_distributed_v06.py test_ski_nl_exchange.py test_semantic_proposition_sd.py test_rsd.py test_rsd_v03.py test_semantic_to_japanese.py
```

Prepare a map `receiver_models.json` with correct **local** paths; these can
be identical for a first integration test, but identical weights do not
demonstrate heterogeneous models:

```json
{
  "LLM-B": {"checkpoint": "vendor/LLM_SEM/model/model-gpu-v0.4.pt", "tokenizer": "vendor/LLM_SEM/model/tokenizer.json"},
  "LLM-C": {"checkpoint": "vendor/LLM_SEM/model/model-gpu-v0.4.pt", "tokenizer": "vendor/LLM_SEM/model/tokenizer.json"}
}
```

Generate the message list from existing derived SDs:

```powershell
python -X utf8 -c "import json; from rsd import make_sd,validate; from ski_nl_exchange import export_message; p=json.load(open('derived_sd.json',encoding='utf-8-sig')); out=[export_message(make_sd(validate(sd),sd['sd_id']),source_llm='LLM-A',target_llm=dst,parent_sd_id=sd['derivation']['parent_id']) for sd,dst in zip(p,['LLM-B','LLM-C'])]; json.dump(out,open('ski_messages_v07.json','w',encoding='utf-8'),ensure_ascii=False,indent=2)"
python semos_real_adapter_v07.py ski_messages_v07.json receiver_models.json results/real_adapter_v07.json --device auto
```

The received local SD contains vector, encoder and tokenizer hashes, receiver
identifier and transfer provenance. **Vector data must stay local**; exchange
of SKI messages and replies uses natural language only.

If Japanese text is out-of-vocabulary for the receiver tokenizer or exceeds
the model context window, the program fails explicitly rather than claiming
successful semantic processing.
