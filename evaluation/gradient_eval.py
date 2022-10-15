from random import Random
from analysis import HolisticAnalyis
from assignment import PDAssignment, RandomAssignment, normalize_priorities
from generator import generate_system, set_utilization
from gradient_descent import *
from examples import *


def test_medium():
    # create a system
    random = Random(123)
    n_flows, t_tasks, n_procs = (4, 5, 3)
    system = generate_system(random,
                             n_flows=n_flows,
                             n_tasks=t_tasks,
                             n_procs=n_procs,
                             utilization=0.8,
                             period_min=100,
                             period_max=100 * 3,
                             deadline_factor_min=0.5,
                             deadline_factor_max=1)

    # prepare GDPA
    proxy = HolisticAnalysisProxy(r_iter=3, max_p=3, w_iter=3, sigmoid_k=20)
    analysis = HolisticAnalyis(reset=False)
    gdpa = GDPA(proxy=proxy, rate=1, delta=0.001, analysis=analysis, verbose=True, iterations=20,
                cost_fn=weighted_invslack)

    # launch GDPA
    system.apply(gdpa)


def test_big():
    # create a system
    random = Random(123)
    system = get_big_system(random)
    set_utilization(system, 0.6)

    # prepare GDPA
    proxy = HolisticAnalysisProxy(r_iter=3, max_p=3, w_iter=3, sigmoid_k=20)
    analysis = HolisticAnalyis(reset=False)
    gdpa = GDPA(proxy=proxy, rate=1, delta=0.001, analysis=analysis, verbose=True, iterations=20,
                cost_fn=weighted_invslack)

    # launch GDPA
    system.apply(gdpa)


if __name__ == '__main__':
    test_big()
