import unittest
from random import Random

from analysis import HolisticAnalyis
from data import to_vector, infer_shape
from examples import get_palencia_system
from generator import generate_system


class MyTestCase(unittest.TestCase):

    def test_to_vector(self):
        system = get_palencia_system()

        system.apply(HolisticAnalyis())
        vector, titles = to_vector(system, (3, 3, 3), normalize=False)
        data = {title:value for title, value in zip(titles, vector)}
        print(data)

        self.assertEqual(vector.shape, titles.shape)
        self.assertEqual(data['period_flow0'], 30)
        self.assertEqual(data['period_flow1'], 40)
        self.assertEqual(data['wcet_flow0_task0_proc0'], 5)
        self.assertEqual(data['wcet_flow0_task0_proc1'], 0)
        self.assertEqual(data['wcet_flow1_task1_proc2'], 10)
        self.assertEqual(data['wcet_flow1_task2_proc1'], 0)
        self.assertEqual(data['priority_flow0_task2_proc1'], 1)
        self.assertEqual(data['priority_flow1_task1_proc1'], 0)
        self.assertEqual(data['label_sched_system'], 0)
        self.assertAlmostEqual(data['label_slack_system'], 0.53333, delta=0.0001)
        self.assertAlmostEqual(data['label_slack_flow0'], 0.53333, delta=0.0001)
        self.assertAlmostEqual(data['label_slack_flow1'], 0.75, delta=0.0001)

    def test_infer_shape(self):
        random = Random(10)
        utilization = 0.5

        systems = [generate_system(random,
                                   n_flows=random.randint(1, 10),
                                   n_tasks=random.randint(1, 10),
                                   n_procs=random.randint(1, 5),
                                   utilization=utilization,
                                   period_min=100, period_max=100 * random.uniform(2.0, 1000.0),
                                   deadline_factor_min=0.5, deadline_factor_max=2)
                   for _ in range(1000)]

        shape = infer_shape(systems)
        self.assertEqual(shape, (10, 10, 5))


if __name__ == '__main__':
    unittest.main()
