import unittest

from analysis import HolisticAnalyis
from data import to_vector
from examples import get_palencia_system


class MyTestCase(unittest.TestCase):
    def test_to_vector(self):
        system = get_palencia_system()

        system.apply(HolisticAnalyis())
        vector, titles = to_vector(system, (2, 3, 3), normalize=False)
        data = {title:value for title, value in zip(titles, vector)}
        print(data)

        self.assertEqual(vector.shape, titles.shape)
        self.assertEqual(data['period_flow1'], 30)
        self.assertEqual(data['period_flow2'], 40)
        self.assertEqual(data['wcet_flow1_a1_cpu1'], 5)
        self.assertEqual(data['wcet_flow1_a1_cpu2'], 0)
        self.assertEqual(data['wcet_flow2_a5_network'], 10)
        self.assertEqual(data['wcet_flow2_a6_cpu2'], 0)
        self.assertEqual(data['priority_flow1_a3_cpu2'], 1)
        self.assertEqual(data['priority_flow2_a5_cpu2'], 0)
        self.assertEqual(data['label_sched_system'], 0)
        self.assertAlmostEqual(data['label_slack_system'], 0.53333, delta=0.0001)
        self.assertAlmostEqual(data['label_slack_flow1'], 0.53333, delta=0.0001)
        self.assertAlmostEqual(data['label_slack_flow2'], 0.75, delta=0.0001)


if __name__ == '__main__':
    unittest.main()
