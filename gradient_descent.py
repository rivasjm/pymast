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


def avg_wcrt(system) -> float:
    return system.avg_flow_wcrt


def weighted_avg_wcrt(system) -> float:
    ws = [math.exp(-f.slack) for f in system.flows]
    num = sum([flow.wcrt*w for flow, w in zip(system.flows, ws)])
    return num/sum(ws)


def calculate_cost(system, analysis, cost_fn) -> float:
    save_wcrt(system)
    analysis.apply(system)
    cost = cost_fn(system)
    restore_wcrt(system)
    return cost


def calculate_gradients(system, proxy, delta=0.01, cost_fn=weighted_avg_wcrt) -> [float]:
    coeffs = []
    for task in system.tasks:
        priority = task.priority
        task.priority = priority - delta
        a = calculate_cost(system, proxy, cost_fn)
        task.priority = priority + delta
        b = calculate_cost(system, proxy, cost_fn)
        task.priority = priority
        diff = (b-a) / (2*delta)
        coeffs.append(diff)
    return coeffs


class GDPA:
    def __init__(self, proxy, iterations=100, rate=0.0001, delta=0.01, analysis=None, verbose=False):
        self.proxy = proxy
        self.iterations = iterations if iterations > 0 else 1
        self.rate = rate if rate > 0 else 0.0001
        self.delta = delta if delta > 0 else 0.01
        self.analysis = analysis
        self.verbose = verbose

    def _iteration_metrics(self, system, cost_fn):
        system.apply(self.analysis if self.analysis else self.proxy)
        cost = cost_fn(system)
        schedulable = system.is_schedulable() if self.analysis else None
        slack = system.slack if self.analysis else None
        return cost, schedulable, slack

    @staticmethod
    def _print_iteration_metrics(iteration, cost, min_cost, schedulable, slack):
        msg = f"iteration={iteration}, cost={cost}, min_cost={min_cost}"
        if slack is not None:
            msg += f", slack={slack}"
        if schedulable is not None:
            msg += f", schedulable={schedulable}"
        print(msg)

    def apply(self, system: System):
        # calculate initial metrics. Uses real analysis if available, proxy otherwise
        cost, schedulable, slack = self._iteration_metrics(system, cost_fn=weighted_avg_wcrt)
        min_cost = cost

        if self.verbose:
            self._print_iteration_metrics(-1, cost, min_cost, schedulable, slack)

        tasks = system.tasks
        for i in range(self.iterations):
            # update priorities using gradient descent and the proxy analysis function
            # I cannot use the real analysis here, because it is not smooth
            coeffs = calculate_gradients(system, self.proxy, self.delta, cost_fn=weighted_avg_wcrt)
            for task, coeff in zip(tasks, coeffs):
                task.priority += task.priority * -coeff * self.rate

            # calculate current metrics. Uses real analysis if available, proxy otherwise
            cost, schedulable, slack = self._iteration_metrics(system, cost_fn=weighted_avg_wcrt)
            if cost < min_cost:
                min_cost = cost
                save_assignment(system)

            if self.verbose:
                self._print_iteration_metrics(i, cost, min_cost, schedulable, slack)

        # restore the best priority assignment found
        restore_assignment(system)



