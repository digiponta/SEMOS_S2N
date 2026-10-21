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
if __name__ == "__main__":
    unittest.main()
