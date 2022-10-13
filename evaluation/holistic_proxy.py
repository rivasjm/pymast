import matplotlib.pyplot as plt
from examples import get_palencia_system, get_barely_schedulable
from analysis import HolisticAnalyis, HolisticAnalyisProxy, reset_wcrt
from assignment import PDAssignment, walk_random_priorities
from random import Random
from generator import generate_system


holistic = HolisticAnalyis(limit_factor=10000)
proxy = HolisticAnalyisProxy()
xs = []
ys = []


def test_cb(system):
    reset_wcrt(system)
    system.apply(holistic)
    x = system.avg_flow_wcrt
    print(x, end=" ")
    xs.append(x)

    reset_wcrt(system)
    system.apply(proxy)
    y = system.avg_flow_wcrt
    print(y)
    ys.append(y)


def proxy_test(system):
    pd = PDAssignment()
    pd.apply(system)

    # normalize priorities
    max_priority = max(map(lambda t: t.priority, system.tasks))
    for t in system.tasks:
        t.priority = t.priority/max_priority

    walk_random_priorities(system, 10, 3, test_cb, seed=5)


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
                             period_max=100*3,
                             deadline_factor_min=0.5,
                             deadline_factor_max=1)
    proxy_test(system)

