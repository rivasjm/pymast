from random import Random
from analysis import HolisticAnalyis
from assignment import PDAssignment, RandomAssignment, normalize_priorities, HOPAssignment, walk_random_priorities
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
                cost_fn=avg_wcrt, over_iterations=10)

    # launch GDPA
    system.apply(gdpa)


def test_barely():
    system = get_barely_schedulable()
    random = Random(2)

    # prepare GDPA
    proxy = HolisticAnalysisProxy(r_iter=5, max_p=5, w_iter=5, sigmoid_k=10)
    analysis = HolisticAnalyis(reset=False)
    hopa = HOPAssignment(analysis, verbose=True, normalize=True)
    rnd = RandomAssignment(normalize=True, random=random)
    pd = PDAssignment(normalize=True)
    adam = Adam(lr=0.1, beta1=0.9, beta2=0.999)
    gdpa = GDPA(proxy=analysis, analysis=analysis, verbose=True, iterations=200,
                cost_fn=invslack, over_iterations=100, initial=rnd, optimizer=adam)

    # launch
    system.apply(gdpa)
    # walk_random_priorities(system, 20, 2, callback=analyze_cb)


def analyze_cb(system):
    analysis = HolisticAnalyis(reset=False)
    analysis.apply(system)
    sched = "SCHEDULABLE" if system.is_schedulable() else "NOT SCHEDULABLE"
    print(f"slack={system.slack} {sched}")


def test_big():
    # create a system
    random = Random(123)
    system = get_big_system(random)
    set_utilization(system, 0.6)

    # prepare GDPA
    proxy = HolisticAnalysisProxy(r_iter=3, max_p=3, w_iter=3, sigmoid_k=20)
    analysis = HolisticAnalyis(reset=False)
    gdpa = GDPA(proxy=proxy, delta=0.001, analysis=analysis, verbose=True, iterations=20,
                cost_fn=weighted_invslack, over_iterations=10)

    # launch GDPA
    system.apply(gdpa)


if __name__ == '__main__':
    test_barely()
