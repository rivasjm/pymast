import numpy as np
from model import System


def successor_matrix(succesors):
    """Builds the successor matrix from a flat successor list. Task id's start from 1"""
    s = np.zeros((succesors.size, succesors.size), dtype=np.int32)
    for i, v in enumerate(succesors):
        if v > -1:
            s[i, v-1] = 1
    return s


def jitter_matrix(smatrix, r):
    """Builds the jitter matrix. Assumes at most 1 successor per task"""
    return smatrix.T @ r


def priority_matrix(priorities):
    """Builds a 3D priority matrix for the given priority scenarios"""
    t, s = priorities.shape
    planes = priorities.ravel(order='F').reshape(s, t, 1)
    P = planes < planes.transpose((0, 2, 1))
    return P


def get_vectors(system: System):
    """Transform a system into vectors. The vectorized analysis is based on these vectors"""
    tasks = system.tasks
    t = len(tasks)
    wcets = np.zeros((t, 1), dtype=np.float32)
    periods = np.zeros((t, 1), dtype=np.float32)
    deadlines = np.zeros((t, 1), dtype=np.float32)
    successors = np.zeros((t, 1), dtype=np.int32)
    mappings = np.zeros((t, 1), dtype=np.object)
    priorities = np.zeros((t, 1), dtype=np.float32)

    for i, task in enumerate(tasks):
        wcets[i] = task.wcet
        periods[i] = task.period
        deadlines[i] = task.flow.deadline
        mappings[i] = task.processor.name
        priorities[i] = task.priority

    return wcets, periods, deadlines, successors, mappings, priorities


def analysis(wcets, periods, deadlines, successors, mappings, priorities, limit=10):
    assert wcets.shape == periods.shape == deadlines.shape == successors.shape == mappings.shape
    assert wcets.shape[1] == 1

    # there are t tasks, and s scenarios
    t, s = priorities.shape

    # 'priorities' has several columns, each column is a priority scenario for the system
    # create a 3D priority matrix, where each plane is a priority matrix for each scenario
    # the objective is to be able to analyze several priority assignments at the same time
    PM = priority_matrix(priorities) * (mappings == mappings.T)

    # the successors' matrix maps, for each task (row), which task is its successor (column)
    # this is a 2D matrix (all scenarios have the same successors mapping)
    S = successor_matrix(successors)

    # initialize response times
    # 3D column vector, each plane for each scenario
    Rmax = np.zeros((s, t, 1), dtype=np.float32)
    Rprev = np.full_like(Rmax, -1.)  # initialized to -1 to bootstrap the main loop

    # a limit on the response times for each task
    # when a task provisional response time reaches its r-limit:
    # - the analysis of its scenario should be stopped
    # - the response time of the affected task and its successors should be set to the limit
    # - the system is therefore deemed non schedulable
    Rlimit = limit * deadlines

    # r mask. 3D column vector
    # when a task response time converges, its value here is set to 0
    # when a task reaches its r-limit, the values for the whole scenario are set to 0
    rmask = np.ones_like(Rmax)

    # stop convergence of response time if all tasks converged, or reached their r-limit
    while rmask.any():
        Rprev = Rmax

        # update jitter matrix with current response times
        # 3D column vector, with jitter for each scenario (plane)
        J = jitter_matrix(S, Rmax)

        # initial activation index
        # TODO: idea to batch together several p iterations
        # define a batch size: how many p values to test at the same time
        # add the batches as additional planes
        # for S scenarios, B batch size: we will have B*S planes
        # I guess I cannot use broadcasting with the STOP vector: expand STOP myself
        # The final +p*wcets in eq (1) cannot be broadcasted either (I guess). Expand this myself too.
        # I need to create a (B*S, tasks, 1) matrix to store the p values for each plane
        p = 1

        # p-limit mask. when a task reaches its p-limit, its bit is set to False here.
        pmask = np.ones_like(J)

        while True:
            # activation index is increased when w is smaller than STOP
            STOP = p * periods

            # initialize W
            W = np.zeros_like(J)
            Wprev = np.full_like(J, -1.)

            # W iteration. this stops when W converges
            while not np.allclose(W, Wprev):
                Wprev = W
                # Eq. (1) of "On the schedulability Analysis for Distributed Hard Real-Time Systems"
                W = np.ceil((PM * W + PM * J.transpose(0, 2, 1)) / periods.T) @ wcets + p * wcets

                # Ignore those tasks that have reached their p-limit already
                W = W * pmask

                # find the provisional response time here
                Rprov = rmask * (W-(p-1)*periods+J)

                # identify the tasks that have reached their r-limit
                # if a task reached its r-limit, set all the r-masks of its scenario to 0
                rmask = rmask * np.all(Rprov < Rlimit, axis=1).reshape((s, 1, 1))

                # also stop the p-iterations if already reached r-limit
                pmask = rmask * pmask

            # once W converges, calculate the response times for this p
            # I can use the last Rprov for this. equation: R = W-(p-1)*periods+J
            R = rmask * Rprov

            # update worst-case response times
            Rmax = np.maximum(R, Rmax)

            # stop the p iterations if all W meet the stopping criteria
            # update the pmask here, and stop when pmask is all zeroes
            pmask = pmask * np.logical_not(W < STOP)
            if not pmask.any():
                break

            # if no stopping criteria, try with next p
            p += 1

        # if a task response time has not changed, sets its bit in the mask to zero
        rmask = rmask * np.logical_not(np.allclose(R, Rprev))

    return Rmax


if __name__ == '__main__':
    # wcets = np.array([5, 2, 20, 5, 10, 10], dtype=np.float32).reshape((-1, 1))
    # periods = np.array([30, 30, 30, 40, 40, 40], dtype=np.float32).reshape((-1, 1))
    # successors = np.array([2, 3, -1, 5, 6, -1], dtype=np.int32).reshape((-1, 1))
    # mappings = np.array([1, 3, 2, 2, 3, 1], dtype=np.int32).reshape((-1, 1))
    # priorities = np.array([10, 1, 1, 10, 10, 1, 10, 1, 10, 1, 10, 1], dtype=np.int32).reshape((-1, 2), order='F')
    # R = analysis(wcets, periods, successors, mappings, priorities)
    # print(R)

    from examples import get_palencia_system
    system = get_palencia_system()
    w, t, d, s, m, p = get_vectors(system)
    print(w)
