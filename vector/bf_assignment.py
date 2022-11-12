import examples
from model import System
import itertools
import vholistic
import random
import numpy as np
import assignment
import time


def scenarios_exec_time():
    # measure vholistic with many priority scenarios
    pd = assignment.PDAssignment()
    system = examples.get_medium_system(random=random.Random(42))
    pd.apply(system)
    tasks = system.tasks
    pop = range(1, len(tasks)+1)
    n = 10000
    prios = np.array([random.sample(pop, len(tasks)) for _ in range(n)]).T
    analysis = vholistic.VectorHolisticAnalysis(limit_factor=1)
    analysis.set_priority_scenarios(prios)
    start = time.time()
    analysis.apply(system)
    finish = time.time()
    print(finish-start)


class BruteForceAssignment:
    def __init__(self, size=10000):
        self.size = size if size > 0 else 1
        self.analysis = vholistic.VectorHolisticAnalysis(limit_factor=1)

    def process(self, system, batch) -> bool:
        scenarios = np.array(batch).T
        self.analysis.clear()
        self.analysis.set_priority_scenarios(scenarios)
        self.analysis.apply(system)
        # now I need to determine if any scenario was schedulable.
        # if a scenario was schedulable, set the priority assignment to the system, and return True
        # otherwise, just return False

        n = len(system.tasks)
        prios = self.analysis.full_priorities
        wcrts = self.analysis.full_response_times
        deadlines = np.array([task.flow.deadline for task in system.tasks]).reshape((n, 1))
        slacks = deadlines-wcrts
        schedulables = np.all(slacks >= 0, axis=0)
        if np.any(schedulables):
            index = np.argmax(schedulables)
            solution = prios[:, index]
            for p, task in zip(solution, system.tasks):
                task.priority = p
            return True
        else:
            return False

    def apply(self, system: System):
        # [[tasks in proc 0], [tasks in proc 1], etc.]
        prios = [range(1, len(p.tasks)+1) for p in system.processors]
        prios = [list(itertools.permutations(p)) for p in prios]

        space = itertools.product(*prios)
        batch = []
        for solution in space:
            batch.append(solution)
            if len(batch) == self.size:
                self.process(system, batch)
                batch.clear()


if __name__ == '__main__':
    system = examples.get_small_system()
    pd = assignment.PDAssignment()
    system.apply(pd)
    bf = BruteForceAssignment(size=3)
    system.apply(bf)




