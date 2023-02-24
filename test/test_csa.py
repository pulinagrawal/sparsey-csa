import unittest
import numpy as np

import sys
from pathlib import Path
sys.path.append(str(Path('.').absolute()))

from src.structure.csa import Macrocolumn
from hypothesis import given, strategies as st

np.random.seed(1111)
INPUT_SIZE = 512

def random_input(size, sparsify=False):
    if sparsify:
        r = np.random.choice(range(size), int(size/10))
        v = np.zeros((size,))
        v[r]=1
    else:
        v = np.random.randint(0, 2, size)
    return v

class TestCSA(unittest.TestCase):

    def test_consecutive_same_input(self):
        m = Macrocolumn()

        input = random_input(512)
        output1 = m.run(input).flatten()
        output2 = m.run(input).flatten()
        
        hamming = sum(abs(output2-output1))
        assert hamming < sum(output1)*.2

    def test_consecutive_diff_input(self):
        m = Macrocolumn()

        input1 = random_input(512)
        output1 = m.run(input1).flatten()

        input2 = random_input(512)
        output2 = m.run(input2).flatten()

        input_hamming = sum(abs(input2-input1))
        output_hamming = sum(abs(output2-output1))

        assert output_hamming > sum(output1)*.8
        assert input_hamming > sum(input1)*.8

    def test_intervaled_same_input(self):
        m = Macrocolumn()

        input = random_input(512)
        output1 = m.run(input).flatten()
        
        for i in range(10):
            input_others = random_input(512)
            _ = m.run(input_others).flatten()

        output2 = m.run(input).flatten()

        hamming = sum(abs(output2-output1))

        print('Reported hamming percentage: '+str(hamming*100/len(output1)))
        print('Reported hamming distance: '+str(hamming))
        print('Reported output cardinality: '+str(sum(output1)))
        print('Reported number minicol: '+str(m.n_minicol))
        assert hamming < len(output1)*.005
