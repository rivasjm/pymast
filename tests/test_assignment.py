import unittest
from examples import *
from assignment import PDAssignment
from model import *


class PDAssignmentTest(unittest.TestCase):

    def test_palencia(self):
        system = get_palencia_system()
        flow1 = system['flow1']
        flow2 = system['flow2']

        #
        system.apply(PDAssignment())

        # test priorities in each processor
        self.assertGreater(flow1['a1'].priority, flow2['a6'].priority)
        self.assertGreater(flow1['a2'].priority, flow2['a5'].priority)
        self.assertLess(flow1['a3'].priority, flow2['a4'].priority)

        # for task in flow1:
        #     print(f"name={task.name}, proc={task.processor.name}, wcet={task.wcet}, priority={task.priority}")
        #
        # for task in flow2:
        #     print(f"name={task.name}, proc={task.processor.name}, wcet={task.wcet}, priority={task.priority}")


if __name__ == '__main__':
    unittest.main()
