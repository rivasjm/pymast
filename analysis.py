from model import Task, System
import math


def higher_priority(task: Task) -> [Task]:
    return [t for t in task.processor.tasks
            if t.priority > task.priority and t != task]


class HolisticAnalyis:
    def apply(self, system: System) -> None:
        while True:
            changed = False

            for task in system.tasks:
                changed |= self._task_analysis(task)

            if not changed:
                break

    def _task_analysis(self, task: Task) -> bool:
        p = 1
        rmax = 0

        while True:
            wip = self._wip(p, task)
            r = self._rip(p, wip, task)

            if r > rmax:
                rmax = r

            # print(f"id={task.name}, wip={wip}, r={r}, rmax={rmax}, p*T={p*task.period}")

            if wip <= p * task.period:
                break

            p += 1

        if math.isclose(task.wcrt, rmax):
            return False
        else:
            task.wcrt = rmax
            return True

    def _rip(self, p: int, wip: float, task: Task) -> float:
        r = wip - (p-1)*task.period + task.jitter
        return r

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