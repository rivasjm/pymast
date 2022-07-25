from random import Random
from math import pow, log, exp
from model import *


def uunifast(random: Random, n_tasks: int, utilization: float) -> [float]:
    sum_u = utilization
    us = []

    for i in range(n_tasks):
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
