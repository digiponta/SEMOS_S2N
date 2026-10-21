import copy
import unittest
from ski_core_v02_exchange import validate_dataset,acknowledge,verify_ack,verify_reply
A={"@context":{"ski":"https://example.org/ski/v0.2#","s2n":"https://example.org/semos-s2n#"},
   "@id":"urn:uuid:aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa","@type":"ski:KnowledgeAssertion",
   "ski:version":1,"ski:issuer":{"@id":"urn:llm:A"},
   "ski:recordedAt":"2026-10-21T17:00:00+09:00","ski:lifecycle":"CANDIDATE",
   "ski:claim":{"ski:subject":{"@id":"urn:concept:GPU"},
      "ski:predicate":{"@id":"urn:relation:supports"},
      "ski:object":{"@id":"urn:concept:parallel"}},
   "s2n:naturalLanguage":"GPUは並列計算を支援する。","ski:conditions":[{"ski:description":"最適化環境"}]}
class ExchangeTests(unittest.TestCase):
    def test_valid_roundtrip(self):
        b=copy.deepcopy(A);b["@id"]="urn:uuid:bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb"
        b["s2n:inReplyTo"]=[A["@id"]];b["s2n:naturalLanguage"]="別の命題で応答する。"
        self.assertTrue(verify_ack([A],acknowledge([A])))
        self.assertTrue(verify_reply([A],[b]))
    def test_attribute_retention(self):
        self.assertEqual(validate_dataset([A])[0]["ski:conditions"][0]["ski:description"],"最適化環境")
    def test_duplicate(self):
        with self.assertRaises(ValueError):validate_dataset([A,A])
    def test_missing_claim(self):
        a=copy.deepcopy(A);del a["ski:claim"]
        with self.assertRaises(ValueError):validate_dataset([a])
    def test_missing_timezone(self):
        a=copy.deepcopy(A);a["ski:recordedAt"]="2026-10-21T17:00:00"
        with self.assertRaises(ValueError):validate_dataset([a])
    def test_bad_ack(self):
        rec=acknowledge([A]);rec["sha256"]="0"*64
        with self.assertRaises(ValueError):verify_ack([A],rec)
    def test_wrong_correlation(self):
        b=copy.deepcopy(A);b["@id"]="urn:uuid:bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb"
        b["s2n:inReplyTo"]=["urn:uuid:cccccccc-cccc-cccc-cccc-cccccccccccc"]
        with self.assertRaises(ValueError):verify_reply([A],[b])
if __name__=="__main__":unittest.main()
