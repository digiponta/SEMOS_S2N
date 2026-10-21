import unittest
from semos_benchmark_v091 import evaluate
def fake_encoder(text,**kwargs):
    group=0 if ("並列" in text or "並行" in text) else 1
    if "逐次" in text or "CPU" in text or "猫" in text: group=1
    v=[0.0]*64;v[group]=1.0
    return {"vector":v,"dimension":64,"encoder_sha256":kwargs["checkpoint"],"tokenizer_sha256":"T"}
MODELS={"LLM-B":{"checkpoint":"B","tokenizer":"T"}}
REF="GPUは並列計算に適している。"
def case(query,category,expected):
    return {"receiver":"LLM-B","reference":REF,"query":query,"category":category,"expected":expected}
class V091Tests(unittest.TestCase):
    def test_metrics(self):
        cases=[case(REF,"exact","KNOWN"),case("GPUは並行演算が得意。","paraphrase","KNOWN"),
               case("GPUは逐次処理のみ。","contradiction","UNKNOWN"),
               case("猫は眠る。","unrelated","UNKNOWN")]
        r=evaluate(cases,MODELS,encoder=fake_encoder)
        self.assertEqual(r["sample_count"],4)
        self.assertEqual(r["balanced_accuracy"],1.0)
        self.assertEqual(r["unknown_reject_rate"],1.0)
    def test_false_accept_counted(self):
        def bad_encoder(text,**kwargs):
            return {"vector":[1.0]+[0.0]*63,"dimension":64,"encoder_sha256":"B","tokenizer_sha256":"T"}
        r=evaluate([case(REF,"exact","KNOWN"),case("猫は眠る。","unrelated","UNKNOWN")],
                   MODELS,encoder=bad_encoder)
        self.assertEqual(r["unknown_reject_rate"],0.0)
        self.assertEqual(r["balanced_accuracy"],0.5)
    def test_invalid_label(self):
        with self.assertRaises(ValueError):evaluate([case("X","exact","MAYBE")],MODELS,encoder=fake_encoder)
    def test_fixed_reference(self):
        cases=[case(REF,"exact","KNOWN"),{**case("X","unrelated","UNKNOWN"),"reference":"別の親命題"}]
        with self.assertRaises(ValueError):evaluate(cases,MODELS,encoder=fake_encoder)
if __name__=="__main__":unittest.main()
