import unittest
from model import *
from analysis import HolisticAnalyis


class HolisticTest(unittest.TestCase):
    def test_palencia(self):
        system = System()

        # 2 cpus + 1 network
        cpu1 = Processor(name="cpu1")
        cpu2 = Processor(name="cpu2")
        network = Processor(name="network")

        # priority levels
        HIGH = 10
        LOW = 1

        # 2 flows
        flow1 = Flow(name="flow1", period=30, deadline=90)
        flow2 = Flow(name="flow2", period=40, deadline=120)

        # tasks
        flow1.add_tasks(
            Task(name="a1", wcet=5,  priority=HIGH, processor=cpu1),
            Task(name="a2", wcet=2,  priority=LOW,  processor=network),
            Task(name="a3", wcet=20, priority=LOW,  processor=cpu2)
        )
        flow2.add_tasks(
            Task(name="a4", wcet=5, priority=HIGH, processor=cpu2),
            Task(name="a5", wcet=10, priority=HIGH, processor=network),
            Task(name="a6", wcet=10, priority=LOW, processor=cpu1)
        )
        system.add_flows(flow1, flow2)

        # analyze
        system.apply(HolisticAnalyis())

        #
        self.assertEqual(flow1.wcrt, 42)
        self.assertEqual(flow2.wcrt, 30)
        self.assertTrue(system.is_schedulable())

        #
        # print(system.processors)


if __name__ == '__main__':
    unittest.main()
