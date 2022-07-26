from model import *


def calculate_priorities(system) -> bool:
    changed = False
    for processor in system.processors:
        tasks = sorted(processor.tasks,
                       key=lambda task: task.local_deadline,
                       reverse=True)

        for i, task in enumerate(tasks):
            if not changed and task.priority != i + 1:
                changed = True
            task.priority = i + 1

    return changed


class PDAssignment:

    def apply(self, system: System):
        PDAssignment.calculate_local_deadlines(system)
        calculate_priorities(system)

    @staticmethod
    def calculate_local_deadlines(system):
        for flow in system:
            sum_wcet = sum(map(lambda t: t.wcet, flow.tasks))
            for task in flow:
                d = task.wcet * flow.deadline / sum_wcet
                task.local_deadline = d


class HOPAssignment:

    def __init__(self, analysis, iterations=50, k_pairs=None, patience=20, verbose=False):
        self.analysis = analysis
        self.k_pairs = k_pairs if k_pairs else HOPAssignment.default_k_pairs()
        self.iterations = iterations
        self.patience = patience
        self.verbose = verbose

    @staticmethod
    def default_k_pairs():
        return [(2.0, 2.0), (1.8, 1.8), (3.0, 3.0), (1.5, 1.5)]

    def apply(self, system: System):
        patience = self.patience
        stop = False
        PDAssignment.calculate_local_deadlines(system)

        for ka, kr in self.k_pairs:
            for i in range(self.iterations):
                if self.verbose:
                    print(f"Iteration={i}, ka={ka}, kr={kr} ", end="")

                changed = calculate_priorities(system)  # update priorities
                patience = patience-1 if not changed else patience

                system.apply(self.analysis)  # update response times
                self.clean_response_times(system)

                if self.verbose:
                    sched = "SCHEDULABLE" if system.is_schedulable() else "NOT SCHEDULABLE"
                    print(f"slack={system.slack} {sched}")

                if system.is_schedulable() or patience <= 0:
                    stop = True
                    break

                self.update_local_deadlines(system, ka, kr)

            if stop:
                break

        self.delete_excesses(system)

    def update_local_deadlines(self, system: System, ka, kr):
        # update excesses with last response times
        for task in system.tasks: self.save_task_excess(task)
        for proc in system.processors: self.save_proc_excess(proc)
        for flow in system.flows: self.save_flow_mex(flow)
        self.save_proc_mex(system)

        # calculate unadjusted local deadlines
        for task in system.tasks:
            self.save_local_deadline(task, ka, kr)

        # adjust local deadlines
        self.adjust_local_deadlines(system)

    def save_local_deadline(self, task: Task, ka, kr):
        mex_pr = task.flow.system.mex_pr
        second = 1 + task.processor.excess/(kr * mex_pr) if kr * mex_pr != 0 else sys.float_info.max
        third = 1 + task.excess/(ka * task.flow.excess) if ka * task.flow.excess != 0 else sys.float_info.max
        task.local_deadline = task.local_deadline * second * third

    def save_task_excess(self, task: Task):
        d = task.local_deadline
        e = 0
        if d <= task.period:
            e = (task.wcrt-d)*task.flow.wcrt/task.flow.deadline
        elif d > task.period:
            e = (task.wcrt+task.jitter-d)*task.flow.wcrt/task.flow.deadline
        task.excess = e

    def save_proc_excess(self, proc: Processor):
        proc.excess = sum([task.excess for task in proc.tasks])

    def save_flow_mex(self, flow: Flow):
        excesses = [abs(task.excess) for task in flow.tasks]
        flow.excess = max(excesses) if len(excesses) > 0 else 0

    def save_proc_mex(self, system: System):
        excesses = [abs(proc.excess) for proc in system.processors]
        system.mex_pr = max(excesses) if len(excesses) > 0 else 0

    def delete_excesses(self, system: System):
        for task in system.tasks:
            if hasattr(task, "excess"): del task.excess
        for flow in system.flows:
            if hasattr(flow, "excess"): del flow.excess
        for proc in system.processors:
            if hasattr(proc, "excess"): del proc.excess
        if hasattr(system, "mex_pr"): del system.mex_pr

    def adjust_local_deadlines(self, system: System):
        for flow in system.flows:
            d_sum = sum([task.local_deadline for task in flow])
            for task in flow:
                task.local_deadline = task.local_deadline * flow.deadline / d_sum

    def clean_response_times(self, system):
        for task in system.tasks:
            if task.wcrt is None:
                task.wcrt = sys.float_info.max
