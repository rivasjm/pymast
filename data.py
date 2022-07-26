import numpy as np
from model import System, Processor


def task_attr_matrix(system: System, shape, procs: [Processor], attr):
    n_flows, n_tasks, n_procs = shape
    matrix = np.zeros(shape)
    matrix_titles = np.zeros(shape, dtype='object')

    for f, flow in enumerate(system.flows):
        if f >= n_flows: break
        for t, task in enumerate(flow.tasks):
            if t >= n_tasks: break
            for p, processor in enumerate(procs):
                if p >= n_procs: break
                matrix[f, t, p] = task.__getattribute__(attr) if task.processor == processor else 0
                matrix_titles[f, t, p] = f"{attr}_{flow.name}_{task.name}_{processor.name}"

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


def to_vector(system: System, shape) -> str:
    n_flows, n_tasks, n_procs = shape
    procs = system.processors

    # matrix with flow periods and deadlines
    periods_deadlines, periods_deadlines_titles = flow_attr_matrix(system, shape, ["period", "deadline"])
    # print(periods_deadlines)

    # wcet 3d matrix
    wcets, wcets_titles = task_attr_matrix(system, shape, procs, "wcet")
    # print(wcets_titles)

    # priority 3d matrix
    priorities, priorities_titles = task_attr_matrix(system, shape, procs, "priority")

    # labels: flow slacks
    flow_slacks, flow_slacks_titles = flow_attr_matrix(system, shape, ["slack"], prefix="label_")
    # print(flow_slacks)





