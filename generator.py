from random import Random
from math import pow, log, exp
from model import *


def uunifast(random: Random, n_tasks: int, utilization: float) -> [float]:
    sum_u = utilization
    us = []

    for i in range(1, n_tasks):
        next_sum_u = sum_u * pow(random.random(), 1 / (n_tasks - i))
        us.append(sum_u - next_sum_u)
        sum_u = next_sum_u

    us.append(sum_u)
    return us


def log_uniform(random: Random, lowest: float, highest: float) -> float:
    r = random.uniform(log(lowest), log(highest))
    return exp(r)


def set_processor_utilization(processor: Processor, utilization: float):
    factor = utilization/processor.utilization
    for task in processor.tasks:
        task.wcet *= factor


def generate_system(random: Random, n_flows, n_procs, n_tasks, utilization,
                    period_min, period_max, deadline_factor_min, deadline_factor_max) -> System:

    procs = [Processor(name=f"proc{i}") for i in range(n_procs)]
    system = System()

    # set the general structure
    for f in range(n_flows):
        period = log_uniform(random, period_min, period_max)
        deadline = random.uniform(period * n_tasks * deadline_factor_min, period * n_tasks * deadline_factor_max)
        flow = Flow(name=f"flow{f}", period=period, deadline=deadline)

        # for now leave the WCET empty
        tasks = [Task(name=f"task{f},{t}", wcet=0, processor=random.choice(procs)) for t in range(n_tasks)]
        flow.add_tasks(*tasks)
        system.add_flows(flow)

    # set the WCET's
    for proc in procs:
        tasks = proc.tasks
        if tasks:
            us = uunifast(random, len(tasks), utilization)
            for task, u in zip(tasks, us):
                task.wcet = u * task.period

    return system


def copy(system: System):
    new_procs = {proc.name: Processor(name=proc.name) for proc in system.processors}
    new_system = System()

    for flow in system:
        new_flow = Flow(name=flow.name, period=flow.period, deadline=flow.deadline)

        for task in flow:
            new_task = Task(name=task.name, wcet=task.wcet, processor=new_procs[task.processor.name],
                            priority=task.priority)
            new_task.wcrt = task.wcrt
            new_flow.add_tasks(task)
        new_system.add_flows(new_flow)

    return new_system
