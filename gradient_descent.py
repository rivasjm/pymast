from analysis import init_wcrt, save_wcrt, restore_wcrt
from assignment import save_assignment, restore_assignment, PDAssignment, normalize_priorities
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


def invslack(system) -> float:
    return max([(flow.wcrt-flow.deadline)/flow.deadline for flow in system.flows])


def weighted_invslack(system) -> float:
    coeffs = [math.exp(-flow.slack) for flow in system.flows]
    num = sum([(flow.wcrt-flow.deadline)/flow.deadline for flow, c in zip(system.flows, coeffs)])
    return num/sum(coeffs)


def calculate_proxy_cost(system, proxy, cost_fn) -> float:
    save_wcrt(system)
    proxy.apply(system)
    cost = cost_fn(system)
    restore_wcrt(system)
    return cost


def calculate_gradients(system, proxy, cost_fn, delta=0.01) -> [float]:
    coeffs = []
    for task in system.tasks:
        priority = task.priority
        task.priority = priority - delta
        a = calculate_proxy_cost(system, proxy, cost_fn)
        task.priority = priority + delta
        b = calculate_proxy_cost(system, proxy, cost_fn)
        task.priority = priority
        diff = (b-a) / (2*delta)
        coeffs.append(diff)
    return coeffs


def avg_parameter_separation(params):
    seps = [abs(params[i+1]-params[i]) for i in range(len(params)-1)]
    return sum(seps)/len(seps)


def keep_max(values: []):
    m = max(values)
    i = values.index(m)
    r = [0]*len(values)
    r[i] = m
    return r


class Adam:
    def __init__(self, lr=0.2, beta1=0.9, beta2=0.999, epsilon=10**-8):
        self.size = None
        self.m = None
        self.v = None
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon

    def step(self, coeffs, iteration) -> [float]:
        if not self.size:
            self.size = len(coeffs)
            self.m = [0]*self.size
            self.v = [0]*self.size

        updates = [0]*self.size

        for i in range(self.size):
            self.m[i] = self.beta1 * self.m[i] + (1 + self.beta1) * coeffs[i]
            self.v[i] = self.beta2 * self.v[i] + (1 + self.beta2) * coeffs[i] ** 2

            me = self.m[i] / (1 - self.beta1 ** iteration)
            ve = self.v[i] / (1 - self.beta2 ** iteration)

            updates[i] = -self.lr*me/(math.sqrt(ve)+self.epsilon)

        return updates


class GDPA:
    def __init__(self, proxy, iterations=100, analysis=None, over_iterations=0,
                 optimizer=Adam(), initial=PDAssignment(normalize=True), cost_fn=invslack, verbose=False):
        self.proxy = proxy
        self.iterations = iterations if iterations > 0 else 1
        self.analysis = analysis
        self.initial = initial
        self.verbose = verbose
        self.cost_fn = cost_fn
        self.over_iterations = over_iterations
        self.optimizer = optimizer
        self.delta = 0

    def _iteration_metrics(self, system):
        system.apply(self.analysis if self.analysis else self.proxy)
        cost = self.cost_fn(system)
        proxy_cost = calculate_proxy_cost(system, proxy=self.proxy, cost_fn=self.cost_fn) if self.analysis else cost
        schedulable = system.is_schedulable() if self.analysis else None
        slack = system.slack if self.analysis else None
        return cost, proxy_cost, schedulable, slack

    @staticmethod
    def _print_iteration_metrics(iteration, cost, proxy_cost, min_cost, schedulable, slack, end="\n"):
        msg = f"{iteration}: proxy={proxy_cost:.2f} [cost={cost:.2f}, best={min_cost:.2f}"
        if slack is not None:
            msg += f", slack={slack:.2f}"
        if schedulable is not None:
            msg += f", schedulable={schedulable}"
        msg += "]"
        print(msg, end=end)

    def apply(self, system: System):
        optimizing = False
        over_iterations = self.over_iterations

        # calculate initial metrics. Uses real analysis if available, proxy otherwise
        self.initial.apply(system)
        cost, proxy_cost, schedulable, slack = self._iteration_metrics(system)
        min_cost = cost

        if self.verbose:
            self._print_iteration_metrics(0, cost, proxy_cost, min_cost, schedulable, slack)

        if schedulable:
            optimizing = True
            over_iterations -= 1
            if over_iterations < 0:
                return

        tasks = system.tasks
        for i in range(1, self.iterations):
            # update priorities using gradient descent and the proxy analysis function
            self.delta = avg_parameter_separation([task.priority for task in system.tasks])
            coeffs = calculate_gradients(system, self.proxy, cost_fn=self.cost_fn, delta=self.delta)
            updates = self.optimizer.step(coeffs, i)
            for task, update in zip(tasks, updates):
                task.priority += update
            normalize_priorities(system)

            # calculate current metrics. Uses real analysis if available, proxy otherwise
            cost, proxy_cost, schedulable, slack = self._iteration_metrics(system)
            if cost < min_cost:
                min_cost = cost
                save_assignment(system)

            if self.verbose:
                self._print_iteration_metrics(i, cost, proxy_cost, min_cost, schedulable, slack)

            if schedulable:
                optimizing = True

            if optimizing:
                over_iterations -= 1

            if optimizing and over_iterations < 0:
                break

        # restore the best priority assignment found
        restore_assignment(system)
