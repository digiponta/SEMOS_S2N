import copy
import unittest
from rsd import make_sd, split, merge, validate, RSDIntegrityError

class ReconstructableSemanticDataTests(unittest.TestCase):
    def test_exact_roundtrip(self):
        text = "GPUは並列計算に適している。\n  長い  空白\t🌸"
        self.assertEqual(validate(make_sd(text)), text)
    def test_merge_unordered(self):
        parent = make_sd("量子センサーは微小な物理量を測定する。")
        left, right = split(parent, 7)
        self.assertEqual(merge([right, left]), parent)
    def test_hierarchical(self):
        parent = make_sd("GPUは並列計算に適している。")
        left, right = split(parent, 5)
        l0, l1 = split(left, 2)
        recovered = merge([merge([l1, l0]), right])
        self.assertEqual(recovered, parent)
    def test_missing_part(self):
        left, right = split(make_sd("日本語ABC"), 3)
        with self.assertRaises(RSDIntegrityError):
            merge([left])
    def test_tamper(self):
        left, right = split(make_sd("日本語ABC"), 3)
        changed = copy.deepcopy(left)
        changed["payload"] = "改竄"
        with self.assertRaises(RSDIntegrityError):
            merge([changed, right])
    def test_mixed_parent(self):
        left, _ = split(make_sd("ABCDE"), 2)
        _, other = split(make_sd("XYZUV"), 2)
        with self.assertRaises(RSDIntegrityError):
            merge([left, other])
    def test_split_guard(self):
        with self.assertRaises(ValueError):
            split(make_sd("ABC"), 0)
    def test_empty_parent(self):
        self.assertEqual(validate(make_sd("")), "")

if __name__ == "__main__":
    unittest.main()
