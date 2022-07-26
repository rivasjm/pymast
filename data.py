import numpy as np
from model import System, Processor


def task_attr_matrix(system: System, shape, procs: [str], attr, default_value=0):
    n_flows, n_tasks, n_procs = shape
    matrix = np.full(shape, default_value, dtype=float)
    matrix_titles = np.zeros(shape, dtype='object')

    for f in range(n_flows):
        for t in range(n_tasks):
            for p in range(n_procs):
                matrix_titles[f, t, p] = f"{attr}_flow{f}_task{t}_proc{p}"
                try:
                    flow = system.flows[f]
                    task = flow.tasks[t]
                    matrix[f, t, p] = task.__getattribute__(attr) if task.processor.name == procs[p] else 0
                except IndexError:
                    continue

    return matrix, matrix_titles


def flow_attr_matrix(system: System, shape, attrs, prefix="", default_value=0):
    n_flows, n_tasks, n_procs = shape
    flows = system.flows
    matrix = np.full((n_flows, len(attrs)), default_value, dtype=float)
    matrix_titles = np.zeros((n_flows, len(attrs)), dtype='object')

    for f in range(n_flows):
        for a, attr in enumerate(attrs):
            matrix_titles[f, a] = f"{prefix}{attr}_flow{f}"
            if f < len(flows):
                matrix[f, a] = flows[f].__getattribute__(attr)

    return matrix, matrix_titles


def flatten(*matrices):
    return np.hstack([matrix.reshape(-1) for matrix in matrices])


def to_vector(system: System, shape, normalize=False) -> str:
    n_flows, n_tasks, n_procs = shape
    # for consistency, sort processors by their name
    procs = [p.name for p in sorted(system.processors, key=lambda p: p.name)]
    d_max = max([flow.deadline for flow in system.flows])  # normalization factor

    # matrix with flow periods and deadlines
    periods_deadlines, periods_deadlines_titles = flow_attr_matrix(system, shape, ["period", "deadline"])
    if normalize:
        periods_deadlines = periods_deadlines/d_max
    # print(periods_deadlines)

    # wcet 3d matrix
    wcets, wcets_titles = task_attr_matrix(system, shape, procs, "wcet")
    if normalize:
        wcets = wcets/d_max
    # print(wcets_titles)

    # priority 3d matrix
    priorities, priorities_titles = task_attr_matrix(system, shape, procs, "priority")
    if normalize:
        priorities = priorities/priorities.max()  # normalize by max, wouldn't make sense to normalize with time (d_max)

    # flow labels: flow slacks
    flow_slacks, flow_slacks_titles = flow_attr_matrix(system, shape, ["slack"], prefix="label_")
    # print(flow_slacks)

    # system labels: system slack, system schedulability, Tmax
    system_labels = np.array([system.slack,
                              0 if system.is_schedulable() else 1,
                              d_max])
    system_labels_titles = np.array(["label_slack_system",
                                     "label_sched_system",
                                     "label_dmax"])

    # vectorize system: flatten all matrices + horizontally append them to form one big vector
    vector = flatten(periods_deadlines, wcets, priorities, flow_slacks, system_labels)
    titles = flatten(periods_deadlines_titles, wcets_titles, priorities_titles, flow_slacks_titles, system_labels_titles)

    return vector, titles


def infer_shape(systems: [System]):
    nflows = max([len(system.flows) for system in systems])
    ntasks = max([len(flow.tasks) for system in systems for flow in system])
    nprocs = max([len(system.processors) for system in systems])
    return nflows, ntasks, nprocs
