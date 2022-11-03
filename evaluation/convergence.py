from model import System
from gradient_descent import invslack, GDPA,Adam
from examples import get_barely_schedulable
from random import Random
from assignment import HOPAssignment, PDAssignment, RandomAssignment
from analysis import HolisticAnalyis, reset_wcrt
import matplotlib.pyplot as plt


lr = 3
delta = 1.5
beta1 = 0.9
beta2 = 0.999
epsilon = 0.1


class Register:
    def __init__(self):
        self.i = 0
        self.n = 1
        self.values = 250*[0]

    def restart(self):
        self.i = 0
        self.n += 1

    def apply(self, system: System):
        metric = invslack(system)
        self.values[self.i] += metric
        self.i += 1


def evaluate():
    # random = Random(2)
    system = get_barely_schedulable()

    analysis = HolisticAnalyis(reset=False, limit_factor=5)
    pd = PDAssignment(normalize=True)
    hopa = HOPAssignment(analysis=analysis, normalize=True, callback=Register(), over_iterations=200)

    gdpa_r = GDPA(proxy=analysis, verbose=False,
                  initial=RandomAssignment(normalize=True),
                  callback=Register(), iterations=200, cost_fn=invslack, analysis=analysis,
                  delta=delta, optimizer=Adam(lr=lr, beta1=beta1, beta2=beta2, epsilon=epsilon))

    gdpa_p = GDPA(proxy=analysis, verbose=False,
                  initial=pd,
                  callback=Register(), iterations=200, cost_fn=invslack, analysis=analysis,
                  delta=delta, optimizer=Adam(lr=lr, beta1=beta1, beta2=beta2, epsilon=epsilon))

    gdpa_h = GDPA(proxy=analysis, verbose=False,
                  initial=HOPAssignment(analysis=analysis, normalize=True, callback=Register(), over_iterations=200),
                  callback=Register(), iterations=200, cost_fn=invslack, analysis=analysis,
                  delta=delta, optimizer=Adam(lr=lr, beta1=beta1, beta2=beta2, epsilon=epsilon))

    hopa.apply(system)
    print(system.is_schedulable())
    reset_wcrt(system)

    gdpa_r.apply(system)
    print(system.is_schedulable())
    reset_wcrt(system)

    gdpa_p.apply(system)
    print(system.is_schedulable())
    reset_wcrt(system)

    gdpa_h.apply(system)
    print(system.is_schedulable())
    reset_wcrt(system)

    plt.plot(hopa.callback.values, label="hopa")
    plt.plot(gdpa_r.callback.values, label="gdpa=r")
    plt.plot(gdpa_p.callback.values, label="gdpa-p")
    plt.plot(gdpa_h.callback.values, label="gdpa-h")
    plt.legend()
    plt.xlim([0, 50])
    plt.show()


if __name__ == '__main__':
    evaluate()