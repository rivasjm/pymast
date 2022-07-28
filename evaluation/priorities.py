from random import Random

from assignment import HOPAssignment, PDAssignment
from generator import generate_system
from analysis import HolisticAnalyis, reset_wcrt
import numpy as np
import matplotlib.pyplot as plt


def hopa_vs_pd():
    random = Random(333)
    holistic = HolisticAnalyis(reset=False)
    pd_assig = PDAssignment()
    hopa_assig = HOPAssignment(analysis=holistic, verbose=False)

    utilizations = np.linspace(0.1, 0.75, 70)
    hopa = []
    pd = []

    for utilization in utilizations:
        # here we store how many systems where schedulable for each utilization
        hopa_scheds = 0
        pd_scheds = 0

        for i in range(100):
            print(f"utilization={utilization}, i={i}", end="")

            system = generate_system(random,
                                     n_flows=random.randint(1, 10),
                                     n_tasks=random.randint(1, 10),
                                     n_procs=random.randint(1, 5),
                                     utilization=utilization,
                                     period_min=100, period_max=100 * random.uniform(2.0, 1000.0),
                                     deadline_factor_min=0.5, deadline_factor_max=2)

            print(" PD", end="")
            reset_wcrt(system)
            system.apply(pd_assig)
            system.apply(holistic)
            if system.is_schedulable():
                pd_scheds += 1

            print(" HOPA")
            reset_wcrt(system)
            system.apply(hopa_assig)
            if system.is_schedulable():
                hopa_scheds += 1

        hopa.append(hopa_scheds)
        pd.append(pd_scheds)

    # chart
    fig, ax = plt.subplots()
    ax.plot(utilizations, pd, color="red")
    ax.plot(utilizations, hopa, color="blue")
    fig.show()


if __name__ == '__main__':
    hopa_vs_pd()