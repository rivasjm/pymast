from model import *


class PDAssignment:

    LOCAL_DEADLINE = "D"

    def apply(self, system: System):
        PDAssignment.calculate_local_deadlines(system)
        PDAssignment.calculate_priorities(system)

        # clean up
        for flow in system:
            for task in flow:
                task.__delattr__(PDAssignment.LOCAL_DEADLINE)

    @classmethod
    def calculate_local_deadlines(cls, system):
        for flow in system:
            sum_wcet = sum(map(lambda t: t.wcet, flow.tasks))
            for task in flow:
                d = task.wcet * flow.deadline / sum_wcet
                task.__setattr__(PDAssignment.LOCAL_DEADLINE, d)

    @classmethod
    def calculate_priorities(cls, system):
        for processor in system.processors:
            tasks = sorted(processor.tasks,
                           key=lambda task: task.__getattribute__(PDAssignment.LOCAL_DEADLINE),
                           reverse=True)
            for i, task in enumerate(tasks):
                task.priority = i+1


class HOPAssignment:

    EXCESS = "EXCESS"

    def __init__(self, pairs, iterations):
        self.pairs = pairs
        self.iterations = iterations
