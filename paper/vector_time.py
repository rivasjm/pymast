from analysis import HolisticAnalyis, reset_wcrt
from vector.vholistic import VectorHolisticAnalysis
from mast.mast_wrapper import MastHolisticAnalysis
import time
import numpy as np
from examples import get_medium_system
from assignment import PDAssignment
from random import Random
import pandas as pd
import matplotlib.pyplot as plt


def step(system, scenarios):
    """
    Measures the execution time of analyzing a number of priority scenarios
    :param system:
    :param scenarios:
    :return: (mast time, holistic time, vector time) tuple
    """
    mast = MastHolisticAnalysis(limit_factor=10)
    hol = HolisticAnalyis(limit_factor=10)
    vec = VectorHolisticAnalysis(limit_factor=10)

    # MAST Holistic: sequentially analyze each priority scenario
    before = time.perf_counter()
    for _ in range(scenarios):
        system.apply(mast)
    mast_time = time.perf_counter() - before

    # Holistic: sequentially analyze each priority scenario
    before = time.perf_counter()
    for _ in range(scenarios):
        system.apply(hol)
    hol_time = time.perf_counter() - before

    # Vector Holistic: analyze all the scenarios in parallel
    before = time.perf_counter()
    t = len(system.tasks)
    if scenarios > 1:
        priorities = np.array([task.priority for task in system.tasks]).reshape((t, 1)).repeat(scenarios-1, axis=1)
        vec.set_priority_scenarios(priorities)
    system.apply(vec)
    vec_time = time.perf_counter() - before

    return np.asarray((mast_time, hol_time, vec_time))


def run(system, name):
    pd = PDAssignment()
    system.apply(pd)

    scenarios = [1, 10, 100, 1000, 10000]
    tools = ["mast", "holistic", "holistic-vector"]
    results = np.zeros(shape=(len(tools), len(scenarios)), dtype=np.float64)

    for i, s in enumerate(scenarios):
        res = step(system, s)
        results[:, i] += res
        save(results, tools, scenarios, name)


def save(results, columns, index, name):
    df = pd.DataFrame(results.T, columns=columns, index=index)
    df.to_excel(f"{name}.xlsx")

    fig, ax = plt.subplots()
    df.plot(ax=ax, xlabel="number of scenarios", ylabel="execution time (s)",
            use_index=True, logy=True, logx=True, figsize=(6, 3))
    fig.tight_layout()
    fig.savefig(f"{name}.png")
    fig.show()


if __name__ == '__main__':
    random = Random(42)
    system = get_medium_system(random, utilization=0.8)
    run(system, "vector-time-medium")


