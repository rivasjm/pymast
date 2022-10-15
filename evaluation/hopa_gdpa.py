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
    gdpa = GDPA(proxy=proxy, rate=1, delta=0.001, verbose=False,
                iterations=20, cost_fn=weighted_invslack)
    gdpa_hopa = GDPA(proxy=proxy, rate=1, delta=0.001, verbose=False,
                     iterations=20, cost_fn=weighted_invslack, initial=HOPAssignment(analysis=analysis, normalize=True))

    random = Random(10)
    pop = 5
    utilizations = np.linspace(0.5, 0.8, pop)
    pds, hopas, gdpas, gdpa_hopas = 0, 0, 0, 0

    for i in range(10):
        print(i)
        template = get_big_system(random)
        systems = create_series(template, utilizations)

        pd_limit = max_schedulable_utilization(systems, pd, analysis)
        pds += pd_limit
        print(f"PD={pd_limit:.2f}")

        hopa_limit = max_schedulable_utilization(systems, hopa, analysis)
        hopas += hopa_limit
        print(f"HOPA={hopa_limit:.2f}")

        gdpa_limit = max_schedulable_utilization(systems, gdpa, analysis)
        gdpas += gdpa_limit
        print(f"GDPA={gdpa_limit:.2f}")

        gdpa_h_limit = max_schedulable_utilization(systems, gdpa_hopa, analysis)
        gdpa_hopas += gdpa_h_limit
        print(f"GDPA-H={gdpa_h_limit:.2f}")

    print(f"PD={pds/20} HOPA={hopas/20} GDPA={gdpas/20} GDPA_HOPA={gdpa_hopas/20}")


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
