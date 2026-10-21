import unittest
from semos_holdout_v093 import summarize
class HoldoutTests(unittest.TestCase):
    def test_abstention_does_not_count_as_correct(self):
        rows=[{"decision":"KNOWN","expected":"KNOWN","correct":True,"category":"exact"},
              {"decision":"REVIEW","expected":"UNKNOWN","correct":False,"category":"unrelated"},
              {"decision":"KNOWN","expected":"UNKNOWN","correct":False,"category":"contradiction"}]
        result=summarize({"version":"v0.9.2","threshold":0.85,"results":rows})
        self.assertAlmostEqual(result["coverage"],2/3)
        self.assertEqual(result["selective_accuracy"],0.5)
        self.assertAlmostEqual(result["strict_accuracy"],1/3)
        self.assertEqual(result["unknown_false_accept_rate"],0.5)
    def test_zero_covered(self):
        row={"decision":"REVIEW","expected":"UNKNOWN","correct":False,"category":"unrelated"}
        result=summarize({"version":"v0.9.2","threshold":0.85,"results":[row]})
        self.assertIsNone(result["selective_accuracy"])
        self.assertEqual(result["coverage"],0)
    def test_missing_label_rejected(self):
        row={"decision":"REVIEW","expected":None,"correct":False,"category":"unrelated"}
        with self.assertRaises(ValueError):summarize({"version":"v0.9.2","threshold":0.85,"results":[row]})
if __name__=="__main__":unittest.main()
