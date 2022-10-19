from analysis import reset_wcrt, HolisticAnalyis
from assignment import PDAssignment, HOPAssignment, RandomAssignment
from gradient_descent import *
from generator import set_utilization
import itertools
import numpy as np
from examples import get_medium_system
from random import Random
from multiprocessing import Pool
from datetime import datetime
import pandas as pd


lrs = [2]
deltas = [1.5]
beta1s = [0.9]
beta2s = [0.999]
epsilons = [0.1]
population = 50
utilization_min = 0.5
utilization_max = 0.9
utilization_steps = 20


def parameters_comparison():
    random = Random(42)
    utilizations = np.linspace(utilization_min, utilization_max, utilization_steps)
    systems = [get_medium_system(random, utilization_min) for _ in range(population)]
    names, _ = zip(*get_assignments(lrs, deltas, beta1s, beta2s, epsilons))
    results = np.zeros((len(names), len(utilizations)))

    i = 0
    for u, utilization in enumerate(utilizations):
        # set utilization to the generated systems
        for system in systems:
            set_utilization(system, utilization)

        #
        with Pool(1) as pool:
            for arr in pool.imap_unordered(step, systems):
                i += 1
                print(f"population={i}")
                results[:, u] += arr
                if i % 5 == 0:
                    overview(i, names, results)

    overview("final", names, results, save=True)


def overview(i, names, results, save=False):
    if save:
        np.savetxt(f"gdpa{i}.csv", results, delimiter=",", header=",".join(utilizations))
    print(f"{datetime.now()} : population={i}")
    for name, res in zip(names, results):
        print(f"  {name} = {res}")


def step(system):
    names, assigs = zip(*get_assignments(lrs, deltas, beta1s, beta2s, epsilons))
    sched_test = get_sched_test()
    results = np.zeros(len(assigs))
    for a, assig in enumerate(assigs):
        print(names[a])
        if achieves_schedulability(system, assig, sched_test):
            results[a] += 1
    return results


def achieves_schedulability(system, assignment, analysis) -> bool:
    reset_wcrt(system)
    system.apply(assignment)
    system.apply(analysis)
    return system.is_schedulable()


def get_assignments(lrs, deltas, beta1s, beta2s, epsilons):
    analysis = HolisticAnalyis(reset=False, limit_factor=5)
    params = itertools.product(lrs, deltas, beta1s, beta2s, epsilons)
    assigs = [("pd", PDAssignment(normalize=True))]

    for lr, delta, beta1, beta2, epsilon in params:
        assig = GDPA(proxy=analysis, verbose=False, initial=RandomAssignment(normalize=True, random=Random(123)),
                     iterations=200, cost_fn=invslack, analysis=analysis, delta=delta,
                     optimizer=Adam(lr=lr, beta1=beta1, beta2=beta2, epsilon=epsilon))
        assigs.append((f"lr={lr} d={delta} b1={beta1} b2={beta2} e={epsilon}", assig))

    return assigs


def get_sched_test():
    return HolisticAnalyis(reset=False, limit_factor=1)


if __name__ == '__main__':
    parameters_comparison()
