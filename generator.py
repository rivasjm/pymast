from random import Random
from math import pow


def uunifast(random: Random, n_tasks: int, utilization: float) -> [float]:
    sum_u = utilization
    us = []

    for i in range(n_tasks):
        next_sum_u = sum_u * pow(random.random(), 1 / (n_tasks - i))
        us.append(sum_u - next_sum_u)
        sum_u = next_sum_u

    us.append(sum_u)
    return us

