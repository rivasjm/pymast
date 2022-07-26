import sys


class System:
    def __init__(self):
        self.flows = list()

    def add_flows(self, *flows):
        self.flows += flows
        for flow in flows:
            flow.system = self
            for task in flow:
                task.processor.system = self

    def __getitem__(self, item):
        if isinstance(item, int):
            return self.flows[item]
        elif isinstance(item, str):
            match =[flow for flow in self.flows if flow.name == item]
            return match[0] if len(match) > 0 else None
        return None

    def apply(self, function):
        function.apply(self)

    @property
    def tasks(self):
        return [task for flow in self.flows for task in flow]

    @property
    def processors(self):
        return {task.processor for flow in self.flows for task in flow}

    def processor(self, name):
        return next((p for p in self.processors if p.name == name), None)

    def is_schedulable(self):
        return all(map(lambda f: f.is_schedulable(), self))

    @property
    def utilization(self):
        us = [proc.utilization for proc in self.processors]
        return sum(us)/len(us)

    @property
    def max_utilization(self):
        us = [proc.utilization for proc in self.processors]
        return max(us)

    @property
    def slack(self):
        slacks = [flow.slack for flow in self.flows]
        return min(slacks) if len(slacks) > 0 else sys.float_info.min


class Processor:
    def __init__(self, name):
        self.system = None
        self.name = name

    def __repr__(self):
        return f"{self.name} ({id(self)})"

    @property
    def tasks(self):
        return [task for flow in self.system for task in flow
                if task.processor == self] if self.system else []

    @property
    def utilization(self):
        u = [task.wcet / task.period for task in self.tasks]
        return sum(u)


class Flow:
    def __init__(self, name, period, deadline):
        self.system = None
        self.name = name
        self.period = period
        self.deadline = deadline
        self.tasks = list()

    def add_tasks(self, *tasks):
        self.tasks += tasks
        for task in tasks:
            task.flow = self

    def __repr__(self):
        return f"{self.name} ({id(self)})"

    @property
    def wcrt(self):
        return self.tasks[-1].wcrt

    @property
    def slack(self):
        if self.wcrt:
            return (self.deadline - self.wcrt)/self.deadline
        else:
            return sys.float_info.min

    def predecessors(self, task):
        i = self.tasks.index(task)
        return [self.tasks[i-1]] if i > 0 else []

    def successors(self, task):
        i = self.tasks.index(task)
        return [self.tasks[i + 1]] if i < len(self.tasks)-1 else []

    def all_successors(self, task):
        i = self.tasks.index(task)
        return self.tasks[i+1:]

    def is_schedulable(self):
        return self.wcrt and self.wcrt <= self.deadline

    def __getitem__(self, item):
        if isinstance(item, int):
            return self.tasks.__getitem__(item)
        elif isinstance(item, str):
            match = [task for task in self.tasks if task.name == item]
            return match[0] if len(match) > 0 else None
        return None


class Task:
    def __init__(self, name: str, wcet: float,
                 processor: Processor = None, priority: int = 0):
        self.flow = None
        self.name = name
        self.wcet = wcet

        self.processor: processor.Processor = processor
        self.priority: int = priority
        self.local_deadline = None
        self.wcrt = None

    def __repr__(self):
        return f"{self.name} ({id(self)})"

    @property
    def utilization(self):
        return self.wcet / self.period

    @property
    def period(self):
        return self.flow.period

    @property
    def all_successors(self):
        return self.flow.all_successors(self)

    @property
    def jitter(self):
        wcrts = list(map(lambda t: t.wcrt, self.flow.predecessors(self)))
        return max(wcrts) if len(wcrts) > 0 else 0

    def copy(self):
        new_task = Task(name=self.name, wcet=self.wcet, processor=None,
                        priority=self.priority)
        new_task.wcrt = self.wcrt
        new_task.local_deadline = self.local_deadline
        return new_task
