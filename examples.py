from random import Random

from generator import generate_system
from model import *


def get_palencia_system() -> System:
    system = System()

    # 2 cpus + 1 network
    cpu1 = Processor(name="cpu1")
    cpu2 = Processor(name="cpu2")
    network = Processor(name="network")

    # priority levels
    HIGH = 10
    LOW = 1

    # 2 flows
    flow1 = Flow(name="flow1", period=30, deadline=90)
    flow2 = Flow(name="flow2", period=40, deadline=120)

    # tasks
    flow1.add_tasks(
        Task(name="a1", wcet=5, priority=HIGH, processor=cpu1),
        Task(name="a2", wcet=2, priority=LOW, processor=network),
        Task(name="a3", wcet=20, priority=LOW, processor=cpu2)
    )
    flow2.add_tasks(
        Task(name="a4", wcet=5, priority=HIGH, processor=cpu2),
        Task(name="a5", wcet=10, priority=HIGH, processor=network),
        Task(name="a6", wcet=10, priority=LOW, processor=cpu1)
    )
    system.add_flows(flow1, flow2)
    return system


def get_barely_schedulable() -> System:
    random = Random(123)
    shape = (4, 5, 3)
    n_flows, t_tasks, n_procs = shape
    system = generate_system(random,
                             n_flows=n_flows,
                             n_tasks=t_tasks,
                             n_procs=n_procs,
                             utilization=0.84,
                             period_min=100,
                             period_max=100*3,
                             deadline_factor_min=0.5,
                             deadline_factor_max=1)
    return system


def get_medium_system(random=Random(), utilization=0.84) -> System:
    n_flows, t_tasks, n_procs = (4, 5, 3)
    system = generate_system(random,
                             n_flows=n_flows,
                             n_tasks=t_tasks,
                             n_procs=n_procs,
                             utilization=utilization,
                             period_min=100,
                             period_max=100*3,
                             deadline_factor_min=0.5,
                             deadline_factor_max=1)
    return system


def get_big_system(random=Random(), utilization=0.84) -> System:
    n_flows, t_tasks, n_procs = (8, 8, 5)
    system = generate_system(random,
                             n_flows=n_flows,
                             n_tasks=t_tasks,
                             n_procs=n_procs,
                             utilization=utilization,
                             period_min=100,
                             period_max=100*3,
                             deadline_factor_min=0.5,
                             deadline_factor_max=1)
    return system