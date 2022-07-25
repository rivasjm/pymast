import unittest
from generator import uunifast, set_processor_utilization, log_uniform
from random import Random
import numpy as np
from examples import get_palencia_system


class GeneratorTest(unittest.TestCase):
    def test_uunifast(self):
        random = Random()
        for n_tasks in range(100):
            for u in np.linspace(0.1, 1.0, num=100):
                us = uunifast(random, n_tasks, u)
                self.assertAlmostEqual(sum(us), u, delta=0.00001)

    def test_set_processor_utilization(self):
        system = get_palencia_system()
        for proc in system.processors:
            for u in np.linspace(0.1, 1.0, num=100):
                set_processor_utilization(proc, u)
                self.assertAlmostEqual(proc.utilization, u, delta=0.00001)

    def test_log_uniform(self):
        random = Random()
        for i in range(1000):
            low = random.uniform(10, (i+1)*10)
            high = low*(i+2)
            value = log_uniform(random, low, high)
            self.assertGreater(value, low)
            self.assertLess(value, high)


if __name__ == '__main__':
    unittest.main()
