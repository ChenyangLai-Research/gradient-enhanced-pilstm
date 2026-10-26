import unittest
import torch
from model import myLSTM


class SeedTest(unittest.TestCase):
    def test_constructor_respects_seed(self):
        torch.manual_seed(1)
        first = myLSTM(23, 32, [33, 16, 1], 1)
        torch.manual_seed(1)
        same = myLSTM(23, 32, [33, 16, 1], 1)
        torch.manual_seed(2)
        different = myLSTM(23, 32, [33, 16, 1], 1)
        self.assertTrue(
            all(
                torch.equal(a, b) for a, b in zip(first.parameters(), same.parameters())
            )
        )
        self.assertFalse(
            all(
                torch.equal(a, b)
                for a, b in zip(first.parameters(), different.parameters())
            )
        )
