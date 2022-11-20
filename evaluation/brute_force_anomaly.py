from random import Random

from analysis import HolisticAnalyis
from examples import get_system, generate_anomaly_system
import numpy as np
from generator import set_utilization
from evaluation.gdpa_eval import achieves_schedulability
from assignment import HOPAssignment
from vector import bf_assignment as bf


def analyze_log(file):
    with open(file) as f:
        for line in f.readlines():
            if has_anomaly(line):
                print(line.rstrip("\n"))


def has_anomaly(line):
    tools = line.split("->")[1].strip()
    if "bf" not in tools and tools:
        return True
    return False


# def generate_anomaly_system():
#     """
#     Anomaly trace: 8-th utilization, system 9 and 23
#     2105-bf-hol-2 0.67(8) 9 	-> hopa gdpa-r gdpa-pd
#     2105-bf-hol-2 0.67(8) 23 	-> hopa gdpa-pd
#     :return:
#     """
#     size = (2, 10, 5)  # flows, tasks/flow, processors
#     population = 50
#     utilization_min = 0.5
#     utilization_max = 0.9
#     utilization_steps = 20
#
#     random = Random(42)
#     utilizations = np.linspace(utilization_min, utilization_max, utilization_steps)
#     systems = [get_system(size, random, balanced=True, name=str(i)) for i in range(population)]
#
#     utilization = utilizations[8]  # 0.67 utilization
#     system = systems[9]  #
#     set_utilization(system, utilization)
#     return system


def priorities_repr(system):
    return " ".join(map(lambda t: str(int(t.priority)), system.tasks))


def evaluate_anomaly():
    system = generate_anomaly_system()
    print(system)

    analysis = HolisticAnalyis(limit_factor=2)
    brute = bf.BruteForceAssignment(size=10000, verbose=True)
    hopa = HOPAssignment(analysis=HolisticAnalyis(reset=False, limit_factor=10), normalize=True, verbose=False)

    brute_sched = achieves_schedulability(system, brute, analysis)
    prios = priorities_repr(system)

    hopa_sched = achieves_schedulability(system, hopa, analysis)

    msg = "ERROR" if hopa_sched and not brute_sched else ""
    print(f"brute={brute_sched} hopa={hopa_sched} {msg} -> {prios}")


def analyze_anomaly():
    system = generate_anomaly_system()
    priorities = [3, 2, 1, 3, 2, 2, 4, 3, 1, 3, 4, 3, 1, 2, 4, 2, 4, 1, 1, 4]
    for p, t in zip(priorities, system.tasks):
        t.priority = p

    analysis = HolisticAnalyis(limit_factor=2, verbose=True)
    analysis.apply(system)


if __name__ == '__main__':
    evaluate_anomaly()
