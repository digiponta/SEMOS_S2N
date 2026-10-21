import copy
import unittest
from semantic_proposition_sd import derive, assemble
from rsd import validate, RSDIntegrityError

A = "GPUは並列計算に適しているので、機械学習に有用である。"
B = "GPUは並列計算に適している。"
C = "GPUは機械学習に有用である。"

class PropositionSDTests(unittest.TestCase):
    def test_independent_japanese_reconstruction(self):
        b, c = derive(A, [B, C], model_id="manual-demo")
        self.assertEqual(validate(b), B)
        self.assertEqual(validate(c), C)
        self.assertNotEqual(B+C, A)
    def test_parent_reconstruction(self):
        b, c = derive(A, [B,C])
        self.assertEqual(validate(assemble([c,b])), A)
    def test_missing_child(self):
        b, c = derive(A,[B,C])
        with self.assertRaises(RSDIntegrityError): assemble([b])
    def test_conflicting_parents(self):
        b, _ = derive(A,[B,C])
        _, c = derive("別の文章",["命題一。","命題二。"])
        with self.assertRaises(RSDIntegrityError): assemble([b,c])
    def test_duplicate(self):
        b, c = derive(A,[B,C])
        with self.assertRaises(RSDIntegrityError): assemble([b,copy.deepcopy(b)])
    def test_tamper_witness(self):
        b, c = derive(A,[B,C])
        b["derivation"]["source_witness"] = "偽の原文"
        with self.assertRaises(RSDIntegrityError): assemble([b,c])
    def test_tamper_proposition(self):
        b,c = derive(A,[B,C])
        c["payload"]="改ざん"
        with self.assertRaises(RSDIntegrityError): assemble([b,c])
    def test_multiple_propositions(self):
        parts = derive(A,[B,C,"GPUは演算装置である。"])
        self.assertEqual(validate(assemble(list(reversed(parts)))), A)

if __name__ == "__main__":
    unittest.main()
