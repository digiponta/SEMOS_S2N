import copy
import unittest
from rsd import make_sd, RSDIntegrityError
from rsd_v03 import pack, unpack, deep_split, reconstruct_leaves, benchmark

class RSDV03Tests(unittest.TestCase):
    def test_compression_exact(self):
        original = make_sd("意味を保持する。\n\t🌸" * 100)
        self.assertEqual(unpack(pack(original)), original)
    def test_empty_payload(self):
        self.assertEqual(unpack(pack(make_sd(""))), make_sd(""))
    def test_bad_transport(self):
        record = pack(make_sd("重要な原文"))
        record["transport"]["data"] = "broken!"
        with self.assertRaises(RSDIntegrityError):
            unpack(record)
    def test_compressed_integrity(self):
        record = pack(make_sd("重要な原文"))
        record["text_sha256"] = "0" * 64
        with self.assertRaises(RSDIntegrityError):
            unpack(record)
    def test_decompression_limit(self):
        record = pack(make_sd("日本語" * 500))
        with self.assertRaises(RSDIntegrityError):
            unpack(record, max_bytes=5)
    def test_deep_four(self):
        original = make_sd("GPUと量子センサーの意味情報を保持する。" * 4)
        fragments = deep_split(original, 4)
        self.assertEqual(reconstruct_leaves(list(reversed(fragments))), original)
    def test_missing_leaf(self):
        parts = deep_split(make_sd("abcdefghijklmnop"), 3)
        with self.assertRaises(RSDIntegrityError):
            reconstruct_leaves(parts[:-1])
    def test_duplicate_leaf(self):
        parts = deep_split(make_sd("abcdefghijklmnop"), 3)
        with self.assertRaises(RSDIntegrityError):
            reconstruct_leaves(parts + [copy.deepcopy(parts[0])])
    def test_tampered_leaf(self):
        parts = deep_split(make_sd("abcdefghijklmnop"), 3)
        parts[0]["payload"] = "X"
        with self.assertRaises(RSDIntegrityError):
            reconstruct_leaves(parts)
    def test_benchmark(self):
        data = benchmark("これは圧縮と復元のための日本語文章。" * 100, 3)
        self.assertTrue(data["exact_reconstruction"])
        self.assertEqual(data["leaves"], 8)

if __name__ == "__main__":
    unittest.main()
