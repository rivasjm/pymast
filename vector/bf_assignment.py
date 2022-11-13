import examples
from model import System
import itertools
import vector.vholistic as vholistic
import random
import numpy as np
import assignment
import time
from assignment import PDAssignment


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


def brute_force_space_size(system):
    import math
    facts = [math.factorial(len(proc.tasks)) for proc in system.processors]
    return math.prod(facts)


class BruteForceAssignment:
    def __init__(self, size=10000, verbose=False):
        self.size = size if size > 0 else 1
        self.verbose = verbose
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
        # first apply PD as initial assignment
        pd = PDAssignment()
        pd.apply(system)

        # flat list of tasks grouped by processor
        proc_tasks = flatten([proc.tasks for proc in system.processors])
        # mapping from proc_tasks to a flat list of tasks according to their flow position
        task_mapping = np.array([proc_tasks.index(task) for task in system.tasks])

        # [[tasks in proc 0], [tasks in proc 1], etc.]
        prios = [range(1, len(p.tasks)+1) for p in system.processors]
        prios = [list(itertools.permutations(p)) for p in prios]

        space = itertools.product(*prios)
        batch = []
        processed = 0
        space_size = brute_force_space_size(system)

        for solution in space:
            # transform solution to a flat list, in flow ordering
            trans = np.array(flatten(solution))[task_mapping].tolist()
            batch.append(trans)

            # process batch if its size limit is reached
            if len(batch) == self.size:
                schedulable = self.process(system, batch)
                processed += len(batch)
                if self.verbose:
                    print(f"Processed {processed}/{space_size} possible solutions")
                batch.clear()
                if schedulable:
                    if self.verbose:
                        print("Schedulable solution found")
                    break

        # process remaining solutions in the batch
        if len(batch) > 0:
            schedulable = self.process(system, batch)
            batch.clear()
            if schedulable and self.verbose:
                print("Schedulable Solution found")


def flatten(list):
    return [x for sublist in list for x in sublist]


if __name__ == '__main__':
    system = examples.get_small_system(utilization=0.85, balanced=True)
    size = brute_force_space_size(system)

    pd = assignment.PDAssignment()
    system.apply(pd)
    bf = BruteForceAssignment(size=20000)
    system.apply(bf)




