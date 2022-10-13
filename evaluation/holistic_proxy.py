import matplotlib.pyplot as plt
import numpy as np

from examples import get_palencia_system, get_barely_schedulable
from analysis import HolisticAnalyis, reset_wcrt
from assignment import PDAssignment, walk_random_priorities, normalize_priorities
from random import Random
from generator import generate_system
from gradient_descent import HolisticAnalysisProxy


holistic = HolisticAnalyis(limit_factor=10000)
proxy = HolisticAnalysisProxy(r_iter=3, max_p=3, w_iter=3, sigmoid_k=50)
xs = []
ys = []


def test_cb(system):
    reset_wcrt(system)
    system.apply(holistic)
    x = system.avg_flow_wcrt
    # print(x, end=" ")
    xs.append(x)

    reset_wcrt(system)
    system.apply(proxy)
    y = system.avg_flow_wcrt
    # print(y)
    ys.append(y)


def proxy_test(system):
    pd = PDAssignment()
    pd.apply(system)
    normalize_priorities(system)
    walk_random_priorities(system, 10, 3, test_cb, seed=5)


def proxy_corr_test(system):
    ks = [1, 10, 50, 100]
    rs = [1, 2, 3]
    ps = [1, 2, 3]
    ws = [1, 2, 3]

    pd = PDAssignment()
    pd.apply(system)
    normalize_priorities(system)

    import itertools
    for k, r, p, w in itertools.product(ks, rs, ps, ws):
        xs.clear()
        ys.clear()
        global proxy
        proxy = HolisticAnalysisProxy(r_iter=r, max_p=p, w_iter=w, sigmoid_k=k)
        walk_random_priorities(system, 10, 3, test_cb, seed=5)
        corr = np.corrcoef(xs, ys)[0 , 1]
        print(f"k={k} r={r} p={p} w={w} -> {corr}")


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
    plt.scatter(xs, ys)
    plt.show()
    proxy_corr_test(system)

