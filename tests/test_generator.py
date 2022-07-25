import unittest
from generator import uunifast
from random import Random
import numpy as np


class GeneratorTest(unittest.TestCase):
    def test_uunifast(self):
        random = Random()
        for n_tasks in range(100):
            for u in np.linspace(0.1, 1.0, num=100):
                us = uunifast(random, n_tasks, u)
                self.assertAlmostEqual(sum(us), u, delta=0.00001)


if __name__ == '__main__':
    unittest.main()
