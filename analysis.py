from model import Task, System
import math


def higher_priority(task: Task) -> [Task]:
    return [t for t in task.processor.tasks
            if t.priority > task.priority and t != task]


def init_wcrt(system: System):
    for task in system.tasks:
        task.wcrt = 0


def reset_wcrt(system: System):
    for task in system.tasks:
        task.wcrt = None


class LimitFactorReachedException(Exception):
    def __init__(self, task, response_time, limit):
        self.task = task
        self.response_time = response_time
        self.limit = limit
        self.message = f"Analysis stopped because provisional response time for task {task.name} (R={response_time}) " \
                       f"reached the limit (limit={limit})"
        super().__init__(self.message)


class HolisticAnalyis:
    def __init__(self, limit_factor = 10, reset=True):
        self.limit_factor = limit_factor
        self.reset = reset

    def apply(self, system: System) -> None:
        init_wcrt(system)

        try:
            while True:
                changed = False
                for task in system.tasks:
                    changed |= self._task_analysis(task)
                if not changed:
                    break

        except LimitFactorReachedException as e:
            print(e.message)
            if self.reset:
                reset_wcrt(system)
            else:
                for task in e.task.all_successors:
                    task.wcrt = e.limit

    def _task_analysis(self, task: Task) -> bool:
        p = 1
        rmax = 0
        while True:
            wip = self._wip(p, task)
            r = wip - (p-1)*task.period + task.jitter
            if r > rmax:
                rmax = r

            if r > task.flow.deadline * self.limit_factor:
                raise LimitFactorReachedException(task, r, task.flow.deadline * self.limit_factor)

            # print(f"id={task.name}, wip={wip}, r={r}, rmax={rmax}, p*T={p*task.period}")
            if wip <= p * task.period:
                break
            p += 1

        if math.isclose(task.wcrt, rmax):
            return False
        else:
            task.wcrt = rmax
            return True

    def _wip(self, p: int, task: Task) -> float:
        w_ini = p * task.wcet
        return self._wi(p, w_ini, task)

    def _wi(self, p: int, w_prev: float, task: Task) -> float:
        hp = higher_priority(task)
        w = sum(map(lambda t: math.ceil((t.jitter + w_prev)/t.period)*t.wcet, hp)) + p*task.wcet

        if math.isclose(w, w_prev):
            return w
        else:
            return self._wi(p, w, task)
