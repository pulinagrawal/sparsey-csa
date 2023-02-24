import unittest
import numpy as np
from tqdm import tqdm

import sys
from pathlib import Path
sys.path.append(str(Path('.').absolute()))

from src.structure.ncsa import SegmentedMacrocolumn
from hypothesis import given, strategies as st
import torch

np.random.seed(1111)
INPUT_SIZE = 512

def random_input(size):
    v = torch.rand((size,))
    return v

class TestSMCSA(unittest.TestCase):

    def test_consecutive_same_input(self):
        m = SegmentedMacrocolumn(2048, 512, .02)

        input = random_input(512)
        output1 = m.run(input).flatten()
        output2 = m.run(input).flatten()
        
        hamming = sum(abs(output2-output1))
        assert hamming < sum(output1)*.2

    def test_consecutive_diff_input(self):
        m = SegmentedMacrocolumn(2048, 512, .02)

        input1 = random_input(512)
        output1 = m.run(input1).flatten()

        input2 = random_input(512)
        output2 = m.run(input2).flatten()

        input_hamming = sum(abs(input2-input1))
        output_hamming = sum(abs(output2-output1))

        assert output_hamming > sum(output1)*.8

    def test_intervaled_same_input(self):
        m = SegmentedMacrocolumn(2048, 512, .02, lr=1.)

        input = random_input(512)
        output1 = m.run(input).flatten()
        
        for i in range(10):
            input_others = random_input(512)
            _ = m.run(input_others).flatten()

        output2 = m.run(input).flatten()

        hamming = sum(abs(output2-output1))

        assert hamming < len(output1)*.005

    def test_high_capacity_intervaled_same_input(self):
        m = SegmentedMacrocolumn(2048, 512, .02)

        input = random_input(512)
        output1 = m.run(input).flatten()
        
        for i in tqdm(range(100)):
            input_others = random_input(512)
            _ = m.run(input_others).flatten()

        output2 = m.run(input).flatten()

        hamming = sum(abs(output2-output1))

        assert hamming < len(output1)*.005