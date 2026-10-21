import unittest
from semos_truth_gate_v092 import proposition, gate, evaluate

class TruthGateV092Tests(unittest.TestCase):
    def test_negation(self):
        self.assertEqual(gate("GPUは並列計算に適している。","GPUは並列計算に適していない。",0.999)["decision"],"UNKNOWN")
    def test_subject(self):
        self.assertEqual(gate("GPUは並列計算に適している。","CPUは並列計算に適している。",0.999)["decision"],"UNKNOWN")
    def test_paraphrase(self):
        self.assertEqual(gate("GPUは並列計算に適している。","GPUは並行演算を効率よく実行できる。",0.95)["decision"],"KNOWN")
    def test_unrelated_review(self):
        self.assertEqual(gate("GPUは並列計算に適している。","猫は眠っている。",0.3)["decision"],"REVIEW")
    def test_low_similarity_review(self):
        self.assertEqual(gate("GPUは並列計算に適している。","GPUは並行演算を効率よく実行できる。",0.3)["decision"],"REVIEW")
    def test_restricted_parser(self):
        self.assertIsNone(proposition("GPUは多くの用途に利用される。"))
        self.assertFalse(proposition("GPUは機械学習には役立たない。")["polarity"])
    def test_fingerprint_guard(self):
        def enc(text,**kw):
            fingerprint="one" if "適している。" in text else "two"
            return {"vector":[1.0]+[0.0]*63,"encoder_sha256":fingerprint,"tokenizer_sha256":"T"}
        row={"receiver":"LLM-B","reference":"GPUは並列計算に適している。",
             "query":"GPUは並列計算に適していない。","expected":"UNKNOWN"}
        with self.assertRaises(ValueError):
            evaluate([row],{"LLM-B":{"checkpoint":"x","tokenizer":"t"}},encoder=enc)
if __name__=="__main__": unittest.main()
