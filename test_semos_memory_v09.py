import unittest
from unittest.mock import patch
from semos_memory_v09 import LocalMemory,evaluate
def v(data,fingerprint="A"):
    return {"vector":data,"dimension":64,"encoder_sha256":fingerprint,"tokenizer_sha256":"T"}
def vec(*coords): return list(coords)+[0.0]*(64-len(coords))
class MemoryV09Tests(unittest.TestCase):
    def test_known_exact(self):
        m=LocalMemory("LLM-B",0.9);m.insert("GPUは速い。",v(vec(1,0)))
        self.assertEqual(m.query("GPUは速い。",v(vec(1,0)))["decision"],"KNOWN")
    def test_unknown_orthogonal(self):
        m=LocalMemory("LLM-B",0.9);m.insert("GPUは速い。",v(vec(1,0)))
        self.assertEqual(m.query("無関係",v(vec(0,1)))["decision"],"UNKNOWN")
    def test_empty(self):
        self.assertEqual(LocalMemory("LLM-B").query("何？",v(vec(1,0)))["decision"],"UNKNOWN")
    def test_duplicate(self):
        m=LocalMemory("LLM-B");self.assertTrue(m.insert("A",v(vec(1,0))))
        self.assertFalse(m.insert("A",v(vec(1,0))))
    def test_mixed_checkpoint_rejected(self):
        m=LocalMemory("LLM-B");m.insert("A",v(vec(1,0)))
        with self.assertRaises(ValueError):m.query("A",v(vec(1,0),"B"))
    def test_zero_vector_rejected(self):
        with self.assertRaises(ValueError):LocalMemory("LLM-B").insert("A",v(vec(0,0)))
    @patch("semos_memory_v09.encode_japanese")
    def test_two_receivers_isolated(self,mock):
        def encode(text,**kw):
            f="A" if kw["checkpoint"]=="A.pt" else "B"
            return v(vec(1,0) if text=="A" else vec(0,1),f)
        mock.side_effect=encode
        model={"LLM-B":{"checkpoint":"A.pt","tokenizer":"T"},
               "LLM-C":{"checkpoint":"B.pt","tokenizer":"T"}}
        cases=[{"receiver":"LLM-B","operation":"teach","text":"A"},
               {"receiver":"LLM-C","operation":"teach","text":"B"},
               {"receiver":"LLM-B","operation":"query","text":"B","expected":"UNKNOWN"},
               {"receiver":"LLM-C","operation":"query","text":"B","expected":"KNOWN"}]
        report=evaluate(cases,model)
        self.assertEqual(report["labeled_query_accuracy"],1.0)
if __name__=="__main__":unittest.main()
