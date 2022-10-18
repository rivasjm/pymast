from analysis import reset_wcrt, HolisticAnalyis
from assignment import PDAssignment, HOPAssignment, RandomAssignment
from gradient_descent import *
import itertools
import numpy as np
from examples import get_medium_system
from random import Random
from multiprocessing import Pool


lrs = [0.01, 0.1, 0.2]
deltas = [0.75, 1, 1.25]
population = 10


def parameters_comparison():
    random = Random(42)
    systems = [get_medium_system(random, 0.6)]
    names, _ = zip(*get_assignments(lrs, deltas))
    results = np.zeros(len(names))

    i = 0
    with Pool(4) as pool:
        for arr in pool.map(step, systems):
            i += 1
            print(f"Population={i}")
            results += arr
            if i % 5 == 0:
                overview(i, names, results)

    overview("TOTAL", names, results)


def overview(i, names, results):
    for res, name in zip(names, results):
        print(f"  {name} = {res}")


def step(system):
    names, assigs = zip(*get_assignments(lrs, deltas))
    sched_test = get_sched_test()
    results = np.zeros(len(assigs))
    for a, assig in enumerate(assigs):
        print(f"{names[a]}")
        if achieves_schedulability(system, assig, sched_test):
            results[a] += 1
    return results


def achieves_schedulability(system, assignment, analysis) -> bool:
    reset_wcrt(system)
    system.apply(assignment)
    system.apply(analysis)
    return system.is_schedulable()


def get_assignments(lrs, deltas):
    analysis = HolisticAnalyis(reset=False, limit_factor=5)
    params = itertools.product(lrs, deltas)
    assigs = []

    for lr, delta in params:
        assig = GDPA(proxy=analysis, verbose=False, initial=RandomAssignment(normalize=True),
                     iterations=200, cost_fn=invslack, analysis=analysis, delta=delta,
                     optimizer=Adam(lr=lr, beta1=0.9, beta2=0.999, epsilon=10 ** -8))
        assigs.append((f"lr={lr} delta={delta}", assig))

    return assigs


def get_sched_test():
    return HolisticAnalyis(reset=False, limit_factor=1)


if __name__ == '__main__':
    parameters_comparison()
