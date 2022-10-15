from assignment import HOPAssignment, PDAssignment
from gradient_descent import *
from analysis import HolisticAnalyis, reset_wcrt
from examples import get_medium_system, get_big_system
from generator import set_utilization, create_series
from random import Random
import numpy as np
import matplotlib.pyplot as plt


def max_utilization():
    proxy = HolisticAnalysisProxy(r_iter=3, max_p=3, w_iter=3, sigmoid_k=20)
    analysis = HolisticAnalyis(reset=False)
    pd = PDAssignment()
    hopa = HOPAssignment(analysis=analysis)
    gdpa = GDPA(proxy=analysis, rate=0.4, delta=0.2, verbose=False,
                iterations=20, cost_fn=invslack, analysis=analysis)
    gdpa_h = GDPA(proxy=analysis, rate=0.4, delta=0.2, verbose=False, analysis=analysis,
                     iterations=20, cost_fn=invslack, initial=HOPAssignment(analysis=analysis, normalize=True))

    random = Random(123)
    n = 20
    utilizations = np.linspace(0.5, 0.90, n)
    pds, hopas, gdpas, gdpahs = n*[0], n*[0], n*[0], n*[0]
    print(utilizations)

    for i, u in enumerate(utilizations):
        for r in range(100):
            system = get_medium_system(random)
            set_utilization(system, u)

            print(f"u={u:0.2f} r={r}")
            if achieves_schedulability(system, pd, analysis):
                pds[i] += 1
            if achieves_schedulability(system, hopa, analysis):
                hopas[i] += 1
            if achieves_schedulability(system, gdpa, analysis):
                gdpas[i] += 1
            if achieves_schedulability(system, gdpa_h, analysis):
                gdpahs[i] += 1

    # chart
    fig, ax = plt.subplots()
    ax.plot(utilizations, pds, color="red")
    ax.plot(utilizations, hopas, color="green")
    ax.plot(utilizations, gdpas, color="blue")
    ax.plot(utilizations, gdpahs, color="black")
    fig.show()


def max_schedulable_utilization(systems, assignment, analysis) -> float:
    utilization = 0
    for system in systems:
        schedulable = achieves_schedulability(system, assignment, analysis)
        if schedulable:
            utilization = system.utilization
        else:
            break
    return utilization


def achieves_schedulability(system, assignment, analysis) -> bool:
    reset_wcrt(system)
    system.apply(assignment)
    system.apply(analysis)
    return system.is_schedulable()


if __name__ == '__main__':
    max_utilization()
