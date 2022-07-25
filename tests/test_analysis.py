import unittest
from model import *
from examples import *
from analysis import HolisticAnalyis


class HolisticTest(unittest.TestCase):
    def test_palencia(self):
        system = get_palencia_system()
        flow1 = system['flow1']
        flow2 = system['flow2']
        cpu1 = next((p for p in system.processors if p.name == "cpu1"), None)
        cpu2 = next((p for p in system.processors if p.name == "cpu2"), None)
        network = next((p for p in system.processors if p.name == "network"), None)

        # analyze
        system.apply(HolisticAnalyis())

        #
        self.assertEqual(flow1.wcrt, 42)
        self.assertEqual(flow2.wcrt, 30)
        self.assertTrue(system.is_schedulable())
        self.assertAlmostEqual(cpu1.utilization, 0.416, delta=0.001)
        self.assertAlmostEqual(cpu2.utilization, 0.791, delta=0.001)
        self.assertAlmostEqual(network.utilization, 0.316, delta=0.001)
        self.assertAlmostEqual(system.max_utilization, 0.791, delta=0.001)
        self.assertAlmostEqual(system.utilization, 0.508, delta=0.001)

        #
        # print(system.processors)


if __name__ == '__main__':
    unittest.main()
