import unittest
from copy import deepcopy
from rsd import make_sd
from ski_nl_exchange import export_message,FORMAT
from ski_dataset_roundtrip import validate_dataset,ack,verify_ack,build_reply,verify_reply
class SKIDatasetRoundtripTests(unittest.TestCase):
    def setUp(self):
        self.A=[export_message(make_sd(t),source_llm="LLM-A",target_llm="LLM-B")
                for t in ("GPUは並列計算に適している。","猫は眠る。")]
    def test_roundtrip_same_format_different_content(self):
        receipt=ack(self.A,"LLM-B")
        B=build_reply(self.A,receipt,"LLM-B",["処理した。","別の結論。"])
        self.assertEqual([m["format"] for m in B],[FORMAT,FORMAT])
        self.assertTrue(verify_reply(self.A,B,"LLM-A"))
        self.assertNotEqual(self.A[0]["text"],B[0]["text"])
    def test_tamper_detected(self):
        corrupted=deepcopy(self.A);corrupted[0]["text"]="改ざん"
        with self.assertRaises(ValueError): validate_dataset(corrupted,"LLM-B")
    def test_wrong_receiver(self):
        with self.assertRaises(ValueError):ack(self.A,"LLM-C")
    def test_ack_tamper(self):
        receipt=ack(self.A,"LLM-B");receipt["dataset_sha256"]="0"*64
        with self.assertRaises(ValueError):verify_ack(self.A,receipt,"LLM-B")
    def test_response_correlation(self):
        receipt=ack(self.A,"LLM-B")
        B=build_reply(self.A,receipt,"LLM-B",["X","Y"])
        B[0]["derivation"]["response_to_message_id"]="invalid"
        with self.assertRaises(ValueError):verify_reply(self.A,B,"LLM-A")
    def test_duplicate_request(self):
        with self.assertRaises(ValueError):ack([self.A[0],self.A[0]],"LLM-B")
if __name__=="__main__":unittest.main()
