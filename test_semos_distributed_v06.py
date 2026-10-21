import copy
import unittest
from semantic_proposition_sd import derive
from semos_distributed_v06 import run
from rsd import make_sd, RSDIntegrityError

A = "GPUは並列計算に適しているので、機械学習に有用である。"
B = "GPUは並列計算に適している。"
C = "GPUは機械学習に有用である。"

class DispatchV06Tests(unittest.TestCase):
    def setUp(self):
        self.parts = derive(A, [B,C])
    def test_two_receivers(self):
        result = run(self.parts, ["LLM-B","LLM-C"])
        self.assertTrue(result["all_receipts_valid"])
        self.assertEqual([x["verified_result_text"] for x in result["results"]],[B,C])
    def test_no_source_witness_transferred(self):
        result = run(self.parts, ["LLM-B","LLM-C"])
        messages = [x["message"] for x in result["results"]]
        self.assertTrue(all(x["derivation"] is None for x in messages))
        self.assertTrue(all("source_witness" not in str(x) for x in messages))
    def test_distinct_local_sd(self):
        result = run(self.parts, ["LLM-B","LLM-C"])
        self.assertEqual(result["results"][0]["receiver_sd"]["receiver_llm"],"LLM-B")
        self.assertNotEqual(result["results"][0]["receiver_sd"]["sd_id"],self.parts[0]["sd_id"])
    def test_adapter_returns_sd(self):
        def adapter(text, llm):
            return make_sd(text + " 処理済")
        result = run(self.parts,["LLM-B","LLM-C"],adapters={"LLM-B":adapter,"LLM-C":adapter})
        self.assertTrue(all(x["verified_result_text"].endswith("処理済") for x in result["results"]))
    def test_adapter_returns_string(self):
        result = run(self.parts,["LLM-B","LLM-C"],adapters={"LLM-B":lambda t,m:t+"!"})
        self.assertEqual(result["results"][0]["verified_result_text"], B+"!")
        self.assertEqual(result["results"][1]["verified_result_text"], C)
    def test_adapter_invalid(self):
        with self.assertRaises(TypeError):
            run(self.parts,["LLM-B","LLM-C"],adapters={"LLM-B":lambda t,m:42})
    def test_duplicate_destinations(self):
        with self.assertRaises(ValueError):
            run(self.parts,["LLM-B","LLM-B"])
    def test_mismatched_lengths(self):
        with self.assertRaises(ValueError):
            run(self.parts,["LLM-B"])
    def test_mixed_parents(self):
        another = derive("別の文章です。",["一。","二。"])[0]
        with self.assertRaises(RSDIntegrityError):
            run([self.parts[0],another],["LLM-B","LLM-C"])
    def test_tamper(self):
        parts = copy.deepcopy(self.parts)
        parts[0]["payload"]="不正"
        with self.assertRaises(RSDIntegrityError):
            run(parts,["LLM-B","LLM-C"])

if __name__=="__main__":
    unittest.main()
