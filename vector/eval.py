from analysis import HolisticAnalyis
from vholistic import analysis
from examples import get_palencia_system
import numpy as np
from timeit import default_timer as timer
from datetime import timedelta


if __name__ == '__main__':
    system = get_palencia_system()
    wcets = np.array([5, 2, 20, 5, 10, 10], dtype=np.float32).reshape((-1, 1))
    periods = np.array([30, 30, 30, 40, 40, 40], dtype=np.float32).reshape((-1, 1))
    successors = np.array([2, 3, -1, 5, 6, -1], dtype=np.int32).reshape((-1, 1))
    mappings = np.array([1, 3, 2, 2, 3, 1], dtype=np.int32).reshape((-1, 1))
    priorities = np.array([10, 1, 1, 10, 10, 1], dtype=np.int32).reshape((-1, 1))

    holistic = HolisticAnalyis()

    t1 = timer()
    holistic.apply(system)
    t2 = timer()
    print(timedelta(t2-t1))

    t1 = timer()
    analysis(wcets, periods, successors, mappings, priorities)
    t2 = timer()
    print(timedelta(t2 - t1))

