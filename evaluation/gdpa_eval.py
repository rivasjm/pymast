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
from functools import partial
import matplotlib.pyplot as plt


lrs = [2, 3]
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

    job = 0
    for u, utilization in enumerate(utilizations):
        # set utilization to the generated systems
        for system in systems:
            set_utilization(system, utilization)

        #
        with Pool(4) as pool:
            func = partial(step, index=u)
            for arr, index in pool.imap_unordered(func, systems):
                job += 1
                results[:, index] += arr
                if job % 25 == 0:
                    print(f"{datetime.now()} : job={job}")
                if job % population == 0:
                    excel("1", names, utilizations, results)
                    chart("1", names, utilizations, results, save=True)

    excel("1", names, utilizations, results)
    chart("1", names, utilizations, results, save=True)


def print_overview(label, names, utilizations, results):
    df = pd.DataFrame(data=results,
                      index=names,
                      columns=utilizations)
    print(df)


def chart(label, names, utilizations, results, save=False):
    # the export version should be transposed, it is the convention to have the continuous data in the columns
    df = pd.DataFrame(data=np.transpose(results),
                      index=utilizations,
                      columns=names)
    fig, ax = plt.subplots()
    df.plot(ax=ax)
    ax.set_ylabel("Schedulable systems")
    ax.set_xlabel("Average utilization")
    if save:
        fig.savefig(f"gdpa_{label}.png")
    plt.show()


def excel(label, names, utilizations, results):
    # the export version should be transposed, it is the convention to have the continuous data in the columns
    df = pd.DataFrame(data=np.transpose(results),
                      index=utilizations,
                      columns=names)
    df.to_excel(f"gdpa_{label}.xlsx")


def step(system, index):
    names, assigs = zip(*get_assignments(lrs, deltas, beta1s, beta2s, epsilons))
    sched_test = get_sched_test()
    results = np.zeros(len(assigs))
    for a, assig in enumerate(assigs):
        # print(names[a])
        if achieves_schedulability(system, assig, sched_test):
            results[a] += 1
    return results, index


def achieves_schedulability(system, assignment, analysis) -> bool:
    reset_wcrt(system)
    system.apply(assignment)
    system.apply(analysis)
    return system.is_schedulable()


def get_assignments(lrs, deltas, beta1s, beta2s, epsilons):
    analysis = HolisticAnalyis(reset=False, limit_factor=5)
    params = itertools.product(lrs, deltas, beta1s, beta2s, epsilons)

    pd = PDAssignment(normalize=True)
    hopa = HOPAssignment(normalize=True, analysis=analysis)

    assigs = [("pd", pd), ("hopa", hopa)]

    for lr, delta, beta1, beta2, epsilon in params:
        assig = GDPA(proxy=analysis, verbose=False, initial=RandomAssignment(normalize=True),
                     iterations=200, cost_fn=invslack, analysis=analysis, delta=delta,
                     optimizer=Adam(lr=lr, beta1=beta1, beta2=beta2, epsilon=epsilon))
        assigs.append((f"gdpa-r [lr={lr} d={delta} b1={beta1} b2={beta2} e={epsilon}]", assig))

        assig = GDPA(proxy=analysis, verbose=False, initial=pd,
                     iterations=200, cost_fn=invslack, analysis=analysis, delta=delta,
                     optimizer=Adam(lr=lr, beta1=beta1, beta2=beta2, epsilon=epsilon))
        assigs.append((f"gdpa-pd [lr={lr} d={delta} b1={beta1} b2={beta2} e={epsilon}]", assig))

        assig = GDPA(proxy=analysis, verbose=False, initial=hopa,
                     iterations=200, cost_fn=invslack, analysis=analysis, delta=delta,
                     optimizer=Adam(lr=lr, beta1=beta1, beta2=beta2, epsilon=epsilon))
        assigs.append((f"gdpa-hopa [lr={lr} d={delta} b1={beta1} b2={beta2} e={epsilon}]", assig))

    return assigs


def get_sched_test():
    return HolisticAnalyis(reset=False, limit_factor=1)


if __name__ == '__main__':
    parameters_comparison()
