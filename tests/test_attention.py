import unittest

import torch

from lynx.attention import attention, official, position_encoding, sequence


class AttentionTests(unittest.TestCase):
    def test_matches_official_movie_attention(self):
        module = official(2)
        actual = attention(module["Q"], module["K"], module["V"], scaled=False)
        expected = module["attention"](module["Q"], module["K"], module["V"])
        for left, right in zip(actual, expected):
            torch.testing.assert_close(left, right)

    def test_mask_weights_and_future_independence(self):
        original = sequence("Lynx", causal=True)
        changed = sequence("Lyn!", causal=True)
        weights = original[-2]
        torch.testing.assert_close(weights.sum(-1), torch.ones(1, 4))
        self.assertEqual(torch.count_nonzero(weights.triu(1)).item(), 0)
        torch.testing.assert_close(original[-1][:, :3], changed[-1][:, :3])
        unmasked = sequence("Lynx", causal=False)
        self.assertTrue(torch.all(unmasked[-2] > 0))
        self.assertFalse(torch.allclose(original[-1], unmasked[-1]))

    def test_position_information_distinguishes_repeated_tokens(self):
        no_position = sequence("aaaa", position="none")[1]
        torch.testing.assert_close(no_position[0, 0], no_position[0, 1])
        for mode in ("learned", "sinusoidal"):
            x = sequence("aaaa", position=mode)[1]
            self.assertFalse(torch.allclose(x[0, 0], x[0, 1]))
        # Also support odd embedding widths in the helper.
        self.assertEqual(position_encoding(4, 7, "sinusoidal").shape, (4, 7))

    def test_shapes_reproducibility_and_gradients(self):
        first = sequence("あA")
        self.assertEqual(first[1].shape, (1, 4, 8))
        torch.testing.assert_close(first[-1], sequence("あA")[-1])
        module = official(6)["Attention"](8, 4)
        x = torch.randn(2, 5, 8, requires_grad=True)
        output, _ = attention(module.W_q(x), module.W_k(x), module.W_v(x), causal=True)
        torch.testing.assert_close(output, module(x))
        output.square().mean().backward()
        self.assertTrue(torch.isfinite(x.grad).all())

    def test_sequence_limits(self):
        for text in ("", "x" * 33):
            with self.assertRaises(ValueError):
                sequence(text)


if __name__ == "__main__":
    unittest.main()
