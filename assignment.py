from model import *


class PDAssignment:

    def apply(self, system: System):
        for flow in system:
            sum_wcet = sum(map(lambda t: t.wcet, flow.tasks))
            for task in flow:
                d = task.wcet * flow.deadline / sum_wcet
                task.__setattr__("D", d)

        for processor in system.processors:
            tasks = sorted(processor.tasks, key=lambda task: task.__getattribute__("D"), reverse=True)
            for i, task in enumerate(tasks):
                task.priority = i+1