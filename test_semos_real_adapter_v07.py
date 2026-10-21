import unittest
from unittest.mock import patch
from rsd import make_sd, validate
from ski_nl_exchange import export_message
from semos_real_adapter_v07 import process_received, run_messages

class RealAdapterV07Tests(unittest.TestCase):
    def setUp(self):
        self.b = export_message(make_sd("GPUは並列計算に適している。", "B"),
                                source_llm="LLM-A", target_llm="LLM-B")
        self.c = export_message(make_sd("GPUは機械学習に有用である。", "C"),
                                source_llm="LLM-A", target_llm="LLM-C")
        self.models = {x: {"checkpoint":"example.pt","tokenizer":"example.json"} for x in ("LLM-B","LLM-C")}
    def fake_encode(self, text, **kwargs):
        return {"dimension":64,"vector":[0.0]*64,"token_count":4, "encoder_sha256":"fixture"}
    @patch("semos_real_adapter_v07.encode_japanese")
    def test_real_adapter_boundary(self, mock):
        mock.side_effect=self.fake_encode
        local, response=process_received(self.b,receiver="LLM-B",checkpoint="example.pt",tokenizer_path="example.json")
        self.assertEqual(local["receiver_semantic"]["dimension"],64)
        self.assertEqual(local["semantic_mode"],"real-llm-sem-encoder")
        self.assertEqual(response["processing_mode"],"real-llm-sem-encoder")
    @patch("semos_real_adapter_v07.encode_japanese")
    def test_multi_receiver(self, mock):
        mock.side_effect=self.fake_encode
        result=run_messages([self.b,self.c],self.models)
        self.assertTrue(result["all_receipts_valid"])
        self.assertEqual(len(result["results"]),2)
        self.assertEqual(mock.call_count,2)
    def test_missing_model(self):
        with self.assertRaises(ValueError):
            run_messages([self.b,self.c],{"LLM-B":self.models["LLM-B"]})
    def test_duplicate_receiver(self):
        with self.assertRaises(ValueError):
            run_messages([self.b,self.b],self.models)
    def test_receiver_mismatch(self):
        with patch("semos_real_adapter_v07.encode_japanese") as mock:
            with self.assertRaises(ValueError):
                process_received(self.b,receiver="LLM-C",checkpoint="example.pt",tokenizer_path="example.json")
            mock.assert_not_called()
    def test_missing_checkpoint(self):
        from semos_real_adapter_v07 import encode_japanese
        try:
            import torch
        except ImportError:
            self.skipTest("torch is not installed")
        with self.assertRaises(FileNotFoundError):
            encode_japanese("日本語",checkpoint="missing-abc.pt",tokenizer_path="missing-xyz.json",device="cpu")

if __name__ == "__main__":
    unittest.main()
