from random import Random
from analysis import HolisticAnalyis
from assignment import PDAssignment, normalize_priorities
from generator import generate_system
from gradient_descent import *


if __name__ == '__main__':
    # create a system
    random = Random(123)
    n_flows, t_tasks, n_procs = (4, 5, 3)
    system = generate_system(random,
                             n_flows=n_flows,
                             n_tasks=t_tasks,
                             n_procs=n_procs,
                             utilization=0.7,
                             period_min=100,
                             period_max=100 * 3,
                             deadline_factor_min=0.5,
                             deadline_factor_max=1)

    # initial priority assignment
    pd = PDAssignment()
    system.apply(pd)
    normalize_priorities(system)

    # prepare GDPA
    proxy = HolisticAnalysisProxy(r_iter=3, max_p=3, w_iter=3, sigmoid_k=50)
    analysis = HolisticAnalyis()
    gdpa = GDPA(proxy=proxy, rate=0.001, delta=0.01, analysis=analysis, verbose=True, iterations=10,
                cost_fn=invslack)

    # launch GDPA
    system.apply(gdpa)
