from analysis import init_wcrt, save_wcrt, restore_wcrt
from assignment import save_assignment, restore_assignment
from model import *
import math


def sigmoid(x, k=50):
    return 1 / (1 + math.exp(-k * x))


def smooth_ceil(x):
    return 1 + x - math.sin(2 * math.pi * x) / (2 * math.pi)


class HolisticAnalysisProxy:
    def __init__(self, r_iter=3, max_p=3, w_iter=3, sigmoid_k=20):
        self.r_iter = r_iter
        self.max_p = max_p
        self.w_iter = w_iter
        self.sigmoid_k = sigmoid_k

    def apply(self, system: System) -> None:
        init_wcrt(system)

        for _ in range(self.r_iter):
            for task in system.tasks:
                self._task_analysis(task)

    def _task_analysis(self, task: Task):
        p = 1
        rmax = 0
        for _ in range(self.max_p):
            wip = self._wi(p, task)
            r = wip - (p-1)*task.period + task.jitter
            if r > rmax:
                rmax = r

            # print(f"id={task.name}, wip={wip}, r={r}, rmax={rmax}, p*T={p*task.period}")
            p += 1

        task.wcrt = rmax

    def _wi(self, p: int, task: Task) -> float:
        wi = p * task.wcet
        tasks = [t for t in task.processor.tasks if t != task]

        for _ in range(self.w_iter):
            wi = sum(map(lambda t: smooth_ceil((t.jitter + wi)/t.period)*t.wcet*self._interference_factor(task, t), tasks)) + p*task.wcet

        return wi

    def _interference_factor(self, task_own, task_other):
        pdiff = task_other.priority - task_own.priority
        factor = sigmoid(pdiff, k=self.sigmoid_k)
        return factor


def proxy_cost(system, proxy) -> float:
    save_wcrt(system)
    proxy.apply(system)
    cost = system.avg_flow_wcrt
    restore_wcrt(system)
    return cost


def calculate_gradients(system, proxy, delta=0.01) -> [float]:
    coeffs = []
    for task in system.tasks:
        priority = task.priority
        task.priority = priority - delta
        a = proxy_cost(system, proxy)
        task.priority = priority + delta
        b = proxy_cost(system, proxy)
        task.priority = priority
        diff = (b-a) / (2*delta)
        coeffs.append(diff)
    return coeffs


def gradient_descent(system, proxy, iter=100, rate=0.0001, delta=0.01, callback=None):
    tasks = system.tasks
    min_cost = proxy_cost(system, proxy)
    save_assignment(system)
    print(f"Iteration=-1, cost={min_cost}")

    for i in range(iter):
        coeffs = calculate_gradients(system, proxy, delta)
        for task, coeff in zip(tasks, coeffs):
            task.priority += task.priority*-coeff*rate

        cost = proxy_cost(system, proxy)
        print(f"Iteration={i}, min_cost={min_cost}, cost={cost}", end=" ")

        if callback:
            callback(system)

        print("")
        if cost < min_cost:
            min_cost = cost
            save_assignment(system)

    restore_assignment(system)
