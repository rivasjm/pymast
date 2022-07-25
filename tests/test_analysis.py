import unittest
from model import *
from analysis import HolisticAnalyis


class HolisticTest(unittest.TestCase):
    def test_palencia(self):
        system = System()

        # 2 cpus + 1 network
        cpu1 = Processor(system=system, name="cpu1")
        cpu2 = Processor(system=system, name="cpu2")
        network = Processor(system=system, name="network")

        # priority levels
        HIGH = 10
        LOW = 1

        # 2 flows
        flow1 = Flow(system, name="flow1", period=30, deadline=90)
        flow2 = Flow(system, name="flow2", period=40, deadline=120)

        # tasks
        flow1.add_tasks(
            Task(flow=flow1, name="a1", wcet=5,  priority=HIGH, processor=cpu1),
            Task(flow=flow1, name="a2", wcet=2,  priority=LOW,  processor=network),
            Task(flow=flow1, name="a3", wcet=20, priority=LOW,  processor=cpu2)
        )
        flow2.add_tasks(
            Task(flow=flow2, name="a4", wcet=5, priority=HIGH, processor=cpu2),
            Task(flow=flow2, name="a5", wcet=10, priority=HIGH, processor=network),
            Task(flow=flow2, name="a6", wcet=10, priority=LOW, processor=cpu1)
        )
        system.add_flows(flow1, flow2)

        # analyze
        system.apply(HolisticAnalyis())

        #
        self.assertEqual(flow1.wcrt, 42)
        self.assertEqual(flow2.wcrt, 30)

        #
        # print(system.processors)


if __name__ == '__main__':
    unittest.main()
