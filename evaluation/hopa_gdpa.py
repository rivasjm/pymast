import os

from assignment import HOPAssignment, PDAssignment, RandomAssignment, clear_assignment
from gradient_descent import *
from analysis import HolisticAnalyis, reset_wcrt
from examples import get_medium_system, get_big_system
from generator import set_utilization, create_series
import random
import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool, Lock, Process
from datetime import datetime
import itertools
from functools import partial
import time
import mast.mast_wrapper as mast


def init(l):
    global lock
    lock = l


def max_utilization():
    steps = 2
    population = 2
    min_u = 0.4
    max_u = 0.5

    names, _ = zip(*get_assignments())
    n_assigs = len(names)
    results = np.zeros((n_assigs, steps))
    utilizations = np.linspace(min_u, max_u, steps).tolist()

    args = [(p, utilizations.index(u), u, get_medium_system, results)
            for u, p in itertools.product(utilizations, range(population))]
    # random.shuffle(args)
    l = Lock()

    # periodic process that makes the charts
    chart_fn = partial(do_chart, results, l, names, utilizations)
    p = Process(target=periodic, args=(chart_fn, 20))
    # p.start()

    with Pool(1, initializer=init, initargs=(l,)) as pool:
        pool.starmap(step, args)
    # p.kill()

    chart_fn()
    print("FINISHED!!")


def periodic(fn, period):
    while period:
        time.sleep(period)
        fn()


def do_chart(results, lock, names, utilizations):
    lock.acquire()
    fig, ax = plt.subplots()
    for a, name in enumerate(names):
        values = results[a]
        ax.plot(utilizations, values, label=name)
    ax.legend()
    fig.show()
    fig.savefig(f"gdpa.png")
    lock.release()


def step(p, i, utilization, system_getter, results):
    # random = Random(f"{p}{utilization}")
    analysis = get_analysis()
    assigs = get_assignments()

    system = system_getter()
    set_utilization(system, utilization)

    print(f"{datetime.now()} p={p} u={utilization}")
    for a, (name, assig) in enumerate(assigs):
        clear_assignment(system)
        if achieves_schedulability(system, assig, analysis):
            lock.acquire()
            results[a, i] += 1
            lock.release()


def get_analysis():
    analysis = HolisticAnalyis(reset=False, limit_factor=5)
    return analysis


def get_assignments():
    # analysis = HolisticAnalyis(reset=False, limit_factor=5)
    analysis = mast.MastOffsetAnalysis(limit_factor=100)
    pd = PDAssignment()
    hopa = HOPAssignment(analysis=analysis)
    adam = Adam(lr=0.1, beta1=0.9, beta2=0.999, epsilon=10 ** -8)
    # gdpa_r = GDPA(proxy=analysis, verbose=False, initial=RandomAssignment(normalize=True),
    #               iterations=200, cost_fn=invslack, analysis=analysis, optimizer=adam)
    gdpa_p = GDPA(proxy=analysis, verbose=False, initial=PDAssignment(normalize=True),
                  iterations=200, cost_fn=invslack, analysis=analysis, optimizer=adam)
    # gdpa_h = GDPA(proxy=analysis, verbose=False, initial=HOPAssignment(analysis=analysis, normalize=True),
    #               iterations=200, cost_fn=invslack, analysis=analysis, optimizer=adam)
    return [("pd", pd), ("hopa", hopa), ("gdpa_p", gdpa_p)]


def achieves_schedulability(system, assignment, analysis) -> bool:
    reset_wcrt(system)
    system.apply(assignment)
    system.apply(analysis)
    return system.is_schedulable()


if __name__ == '__main__':
    max_utilization()
