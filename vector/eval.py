from analysis import HolisticAnalyis
from vholistic import analysis, get_vectors
from examples import get_palencia_system
import numpy as np
from timeit import default_timer as timer
from datetime import timedelta
from generator import set_utilization


def limit_system():
    holistic = HolisticAnalyis(verbose=True, limit_factor=2, reset=False)

    system = get_palencia_system()
    set_utilization(system, 0.8)
    print(system.utilization)
    t1, t2, t3, t4, t5, t6 = system.tasks
    t1.priority, t2.priority, t3.priority, t4.priority, t5.priority, t6.priority = 1, 1, 1, 2, 2, 2
    system.apply(holistic)
    print([t.wcrt for t in system.tasks])
    return system


if __name__ == '__main__':
    # system = get_palencia_system()
    # wcets = np.array([5, 2, 20, 5, 10, 10], dtype=np.float32).reshape((-1, 1))
    # periods = np.array([30, 30, 30, 40, 40, 40], dtype=np.float32).reshape((-1, 1))
    # deadlines = np.array([30, 30, 30, 40, 40, 40], dtype=np.float32).reshape((-1, 1))
    # successors = np.array([2, 3, -1, 5, 6, -1], dtype=np.int32).reshape((-1, 1))
    # mappings = np.array([1, 3, 2, 2, 3, 1], dtype=np.int32).reshape((-1, 1))
    # priorities = np.array([10, 1, 1, 10, 10, 1], dtype=np.int32).reshape((-1, 1))
    #
    # holistic = HolisticAnalyis()
    #
    # t1 = timer()
    # holistic.apply(system)
    # t2 = timer()
    # print(timedelta(t2-t1))
    #
    # t1 = timer()
    # analysis(wcets, periods, deadlines, successors, mappings, priorities)
    # t2 = timer()
    # print(timedelta(t2 - t1))

    system = limit_system()

    p1 = np.array([2, 1, 1, 2, 2, 1]).reshape((6, 1))
    w, t, d, s, m, p2 = get_vectors(system)
    p = np.hstack((p1, p2))

    r = analysis(w, t, d, s, m, p2, limit=2)
    # print(r)

