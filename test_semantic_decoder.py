import unittest
import torch
from semantic_decoder import SemanticDecoder

class DecoderTests(unittest.TestCase):
    def test_forward_shape(self):
        model = SemanticDecoder(vocab_size=18, semantic_dim=64, hidden_size=16, layers=2)
        result = model(torch.randn(3,64), torch.randint(0,18,(3,7)))
        self.assertEqual(tuple(result.shape), (3,7,18))

    def test_gradient(self):
        model = SemanticDecoder(vocab_size=15, hidden_size=16)
        logits = model(torch.randn(2,64), torch.randint(0,15,(2,5)))
        logits.sum().backward()
        self.assertIsNotNone(model.initial.weight.grad)
        self.assertGreater(float(model.initial.weight.grad.abs().sum()), 0)

    def test_dimension_guard(self):
        model = SemanticDecoder(vocab_size=12)
        with self.assertRaises(ValueError):
            model(torch.randn(1,32), torch.ones(1,4,dtype=torch.long))

    def test_encoder_not_needed_for_decoder_shape(self):
        model = SemanticDecoder(vocab_size=10,hidden_size=16)
        self.assertEqual(model.semantic_dim,64)

if __name__ == "__main__":
    unittest.main()
