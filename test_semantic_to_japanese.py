import unittest
from semantic_to_japanese import convert, ConversionError

class TestConversion(unittest.TestCase):
    def test_source_text(self):
        self.assertEqual(convert({"text": "量子センサーとは高感度な装置。", "vector": [0.1]}), "量子センサーとは高感度な装置。")
    def test_proposition(self):
        self.assertEqual(convert({"subject":"GPU", "value":"高速"}), "GPUは、高速である。")
    def test_relation(self):
        self.assertEqual(convert({"subject":"文学", "relation":"includes", "object":"数学"}), "文学は、数学を含む。")
    def test_vector_rejected(self):
        with self.assertRaises(ConversionError):
            convert({"vector": [0.5]})
    def test_unrecognized_predicate(self):
        with self.assertRaises(ConversionError):
            convert({"subject":"A","predicate":"causes","object":"B"})
    def test_bom_json_cli(self):
        import subprocess
        import sys
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "semantic.json"
            path.write_text('{"subject":"GPU","value":"高速"}', encoding="utf-8-sig")
            cmd = [sys.executable, str(Path(__file__).with_name("semantic_to_japanese.py")), str(path)]
            completed = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8")
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertEqual(completed.stdout.strip(), "GPUは、高速である。")

    def test_rsd_v02_exact(self):
        from rsd import make_sd
        original = "  GPUは並列計算に適している。\n"
        self.assertEqual(convert(make_sd(original)), original)

    def test_rsd_v02_reject_tamper(self):
        from rsd import make_sd
        sd = make_sd("原文")
        sd["payload"] = "改ざん"
        with self.assertRaises(ConversionError):
            convert(sd)

if __name__ == "__main__":
    unittest.main()
