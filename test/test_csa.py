import unittest
import numpy as np

import sys
from pathlib import Path
sys.path.append(str(Path('.').absolute()))

from src.structure.csa import Macrocolumn
from hypothesis import given, strategies as st

class TestCSA(unittest.TestCase):

    def test_same_input(self):
        m = Macrocolumn()
        input = np.random.randint(0, 2, 512)

        output1 = m.run(input).flatten()

        output2 = m.run(input).flatten()

        hamming = sum(abs(output2-output1))

        print('Reported hamming percentage: '+str(hamming*100/sum(output1)))
        print('Reported hamming distance: '+str(hamming))
        print('Reported output cardinality: '+str(sum(output1)))
        print('Reported number minicol: '+str(m.n_minicol))
        assert hamming < sum(output1)*.2
