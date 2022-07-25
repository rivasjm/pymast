
class System:
    def __init__(self):
        self.flows = list()

    def add_flows(self, *flows):
        self.flows += flows

    def __getitem__(self, item):
        return self.flows[item]

    def apply(self, function):
        function.apply(self)

    @property
    def tasks(self):
        return [task for flow in self.flows for task in flow]

    @property
    def processors(self):
        return {task.processor for flow in self.flows for task in flow}


class Processor:
    def __init__(self, system: System, name):
        self.system = system
        self.name = name

    def __eq__(self, other):
        if isinstance(other, Processor):
            return self.name == other.name
        return False

    def __hash__(self):
        return self.name.__hash__()

    def __repr__(self):
        return f"Processor({self.name})"

    @property
    def tasks(self):
        return [task for flow in self.system for task in flow
                if task.processor == self]


class Flow:
    def __init__(self, system, name, period, deadline):
        self.system = system
        self.name = name
        self.period = period
        self.deadline = deadline
        self.tasks = list()

    def add_tasks(self, *tasks):
        self.tasks += tasks

    @property
    def wcrt(self):
        return self.tasks[-1].wcrt

    def predecessors(self, task):
        i = self.tasks.index(task)
        return [self.tasks[i-1]] if i > 0 else []

    def successors(self, task):
        i = self.tasks.index(task)
        return [self.tasks[i + 1]] if i < len(self.tasks)-1 else []

    def __getitem__(self, item):
        if isinstance(item, int):
            return self.tasks.__getitem__(item)
        elif isinstance(item, str):
            match = [task for task in self.tasks if task.name == item]
            return match[0] if len(match) > 0 else None


class Task:
    def __init__(self, flow: Flow, name: str, wcet: float,
                 processor: Processor = None, priority: int = 0):
        self.flow = flow
        self.name = name
        self.wcet = wcet

        self.processor: processor.Processor = processor
        self.priority: int = priority
        self.wcrt = 0

    @property
    def period(self):
        return self.flow.period

    @property
    def jitter(self):
        wcrts = list(map(lambda t: t.wcrt, self.flow.predecessors(self)))
        return max(wcrts) if len(wcrts) > 0 else 0
