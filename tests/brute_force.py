import vector.bf_assignment as bf
from assignment import HOPAssignment
from analysis import HolisticAnalyis, reset_wcrt
from examples import get_system
from random import Random
from generator import set_utilization
from evaluation.gdpa_eval import achieves_schedulability


def find_system_problem():
    random = Random(42)
    analysis = HolisticAnalyis(limit_factor=1)
    brute = bf.BruteForceAssignment(size=10000, verbose=True)
    hopa = HOPAssignment(analysis=HolisticAnalyis(reset=False, limit_factor=10), normalize=True, verbose=True)

    size = (2, 10, 5)  # flows, tasks/flow, processors
    systems = [get_system(size, random, balanced=True) for _ in range(50)]
    for s in systems:
        set_utilization(s, 0.5)

    # for system in systems:
    #     print(" ".join(map(lambda t: str(t.wcet), system.tasks)))

    for n, system in enumerate(systems):
        # if n < 44:
        #     continue

        print(f"ITERATION {n} : {system.flows[0].period}, {system.flows[1].period}")
        print("###### BRUTE FORCE ######")
        brute_sched = achieves_schedulability(system, brute, analysis)

        # print("###### HOPA ######")
        # hopa_sched = achieves_schedulability(system, hopa, analysis)

        # if brute_sched != hopa_sched:
        #     print(f"PROBLEMA EN SISTEMA {n} : {system.flows[0].period}, {system.flows[1].period}")
        #     break


def brute_force_problem():
    system = get_problem_system()
    analysis = HolisticAnalyis(limit_factor=1)
    brute = bf.BruteForceAssignment(size=10000, verbose=True)
    hopa = HOPAssignment(analysis=HolisticAnalyis(reset=False, limit_factor=10), normalize=True, verbose=True)
    brute_sched = achieves_schedulability(system, brute, analysis)
    hopa_sched = achieves_schedulability(system, hopa, analysis)
    if brute_sched != hopa_sched:
        print("problema confirmado")


def get_problem_system():
    random = Random(42)
    size = (2, 10, 5)  # flows, tasks/flow, processors
    systems = [get_system(size, random, balanced=True) for _ in range(50)]
    system = systems[44]
    set_utilization(system, 0.5)
    return system


if __name__ == '__main__':
    find_system_problem()
