import copy
import unittest
from semantic_proposition_sd import derive
from ski_nl_exchange import export_message, import_message, receipt, validate_receipt
from rsd import validate, RSDIntegrityError

class SKINaturalLanguageTests(unittest.TestCase):
    def setUp(self):
        self.source = "GPUは並列計算に適しているので、機械学習に有用である。"
        self.parts = derive(self.source, ["GPUは並列計算に適している。", "GPUは機械学習に有用である。"])
    def test_independent_distributed_roundtrip(self):
        for sd, target in zip(self.parts, ["LLM-B", "LLM-C"]):
            message = export_message(sd, source_llm="LLM-A", target_llm=target, parent_sd_id=sd["derivation"]["parent_id"])
            destination = import_message(message, receiver_llm=target)
            self.assertEqual(validate(destination), validate(sd))
            self.assertEqual(destination["origin"]["source_sd_id"], sd["sd_id"])
            self.assertEqual(validate_receipt(message, receipt(destination)), validate(sd))
    def test_target_guard(self):
        message = export_message(self.parts[0], source_llm="LLM-A", target_llm="LLM-B")
        with self.assertRaises(ValueError): import_message(message, receiver_llm="LLM-C")
    def test_text_tamper(self):
        message = export_message(self.parts[0], source_llm="LLM-A", target_llm="LLM-B")
        message["text"] = "変更"
        with self.assertRaises(RSDIntegrityError): import_message(message, receiver_llm="LLM-B")
    def test_message_id_tamper(self):
        message = export_message(self.parts[0], source_llm="LLM-A", target_llm="LLM-B")
        message["message_id"] = "wrong"
        with self.assertRaises(RSDIntegrityError): import_message(message, receiver_llm="LLM-B")
    def test_receipt_tamper(self):
        message = export_message(self.parts[0], source_llm="LLM-A", target_llm="LLM-B")
        local = import_message(message, receiver_llm="LLM-B")
        result = receipt(local)
        result["text"] = "変更"
        with self.assertRaises(RSDIntegrityError): validate_receipt(message, result)
    def test_receipt_swapped(self):
        msg1 = export_message(self.parts[0], source_llm="LLM-A", target_llm="LLM-B")
        msg2 = export_message(self.parts[1], source_llm="LLM-A", target_llm="LLM-C")
        result = receipt(import_message(msg1, receiver_llm="LLM-B"))
        with self.assertRaises(RSDIntegrityError): validate_receipt(msg2, result)
    def test_independent_provenance(self):
        sd = self.parts[1]
        msg = export_message(sd, source_llm="LLM-A", target_llm="LLM-C", parent_sd_id=sd["derivation"]["parent_id"])
        reconstructed = import_message(msg, receiver_llm="LLM-C")
        self.assertEqual(reconstructed["origin"]["parent_sd_id"], sd["derivation"]["parent_id"])

if __name__ == "__main__":
    unittest.main()
