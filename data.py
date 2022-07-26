import numpy as np
from model import System, Processor


def task_attr_matrix(system: System, shape, procs: [str], attr):
    n_flows, n_tasks, n_procs = shape
    matrix = np.zeros(shape)
    matrix_titles = np.zeros(shape, dtype='object')

    for f, flow in enumerate(system.flows):
        if f >= n_flows: break
        for t, task in enumerate(flow.tasks):
            if t >= n_tasks: break
            for p, processor_name in enumerate(procs):
                if p >= n_procs: break
                matrix[f, t, p] = task.__getattribute__(attr) if task.processor.name == processor_name else 0
                matrix_titles[f, t, p] = f"{attr}_{flow.name}_{task.name}_{processor_name}"

    return matrix, matrix_titles


def flow_attr_matrix(system: System, shape, attrs, prefix=""):
    n_flows, n_tasks, n_procs = shape
    matrix = np.zeros((n_flows, len(attrs)))
    matrix_titles = np.zeros((n_flows, len(attrs)), dtype='object')

    for f, flow in enumerate(system.flows):
        if f >= n_flows: break
        for a, attr in enumerate(attrs):
            matrix[f, a] = flow.__getattribute__(attr)
            matrix_titles[f, a] = f"{prefix}{attr}_{flow.name}"

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
