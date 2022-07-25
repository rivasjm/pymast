from random import Random
from math import pow


def uunifast(random: Random, n_tasks: int, utilization: float) -> [float]:
    sum = utilization
    us = []

    for i in range(n_tasks):
        next_sum = sum * pow(random.random(), 1 / (n_tasks - i))
        us.append(sum - next_sum)
        sum = next_sum

    us.append(sum)
    return us