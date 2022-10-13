import matplotlib.pyplot as plt
import numpy as np

from examples import get_palencia_system, get_barely_schedulable
from analysis import HolisticAnalyis, reset_wcrt
from assignment import PDAssignment, walk_random_priorities, normalize_priorities
from random import Random
from generator import generate_system
from gradient_descent import *


def callback(system):
    holistic = HolisticAnalyis()
    system.apply(holistic)
    art = system.avg_flow_wcrt
    print(f"art={art}", end="")


if __name__ == '__main__':
    random = Random(123)
    shape = (4, 5, 3)
    n_flows, t_tasks, n_procs = shape
    system = generate_system(random,
                             n_flows=n_flows,
                             n_tasks=t_tasks,
                             n_procs=n_procs,
                             utilization=0.7,
                             period_min=100,
                             period_max=100 * 3,
                             deadline_factor_min=0.5,
                             deadline_factor_max=1)

    pd = PDAssignment()
    pd.apply(system)
    normalize_priorities(system)

    proxy = HolisticAnalysisProxy(r_iter=3, max_p=3, w_iter=3, sigmoid_k=50)
    gradient_descent(system, proxy, rate=0.001, delta=0.1, callback=callback)
