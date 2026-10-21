import unittest
from unittest.mock import patch
from rsd import make_sd
from ski_nl_exchange import export_message
from semos_heterogeneous_v08 import evaluate

class HeterogeneousV08Tests(unittest.TestCase):
    def setUp(self):
        self.messages = [
            export_message(make_sd("GPUは並列計算に適している。","B"),source_llm="LLM-A",target_llm="LLM-B"),
            export_message(make_sd("GPUは機械学習に有用である。","C"),source_llm="LLM-A",target_llm="LLM-C"),
        ]
        self.models={k:{"checkpoint":"mock.pt","tokenizer":"mock.json"} for k in ("LLM-B","LLM-C")}
    @patch("semos_real_adapter_v07.encode_japanese")
    def test_distinct_fingerprints(self,mock):
        def enc(text,**kw):
            suffix="b" if "並列" in text else "c"
            return {"dimension":64,"token_count":5,"vector":[0.5]*64,
                    "encoder_sha256":suffix*64,"tokenizer_sha256":"t"*64}
        mock.side_effect=enc
        result=evaluate(self.messages,self.models)
        self.assertTrue(result["distinct_model_fingerprints"])
        self.assertTrue(result["all_transport_exact"])
        self.assertTrue(result["all_replies_exact"])
        self.assertEqual(result["semantic_equivalence"],"not-evaluated")
    @patch("semos_real_adapter_v07.encode_japanese")
    def test_identical_rejected(self,mock):
        mock.return_value={"dimension":64,"token_count":5,"vector":[1.0]*64,
                         "encoder_sha256":"a"*64,"tokenizer_sha256":"t"*64}
        with self.assertRaises(ValueError):
            evaluate(self.messages,self.models)
    @patch("semos_real_adapter_v07.encode_japanese")
    def test_identical_smoke_mode(self,mock):
        mock.return_value={"dimension":64,"token_count":5,"vector":[1.0]*64,
                         "encoder_sha256":"a"*64,"tokenizer_sha256":"t"*64}
        result=evaluate(self.messages,self.models,require_distinct=False)
        self.assertFalse(result["distinct_model_fingerprints"])
        self.assertEqual(result["model_count"],2)

if __name__=="__main__":
    unittest.main()
