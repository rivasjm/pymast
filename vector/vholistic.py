import numpy as np


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


def analysis(wcets, periods, successors, mappings, priorities):
    assert wcets.shape == periods.shape == successors.shape == mappings.shape
    assert wcets.shape[1] == 1

    # there are t tasks, and s scenarios
    t, s = priorities.shape

    # 'priorities' has several columns, each column is a priority scenario for the system
    # create a 3D priority matrix, where each plane is a priority matrix for each scenario
    # the objective is to be able to analyze several priority assignments at the same time
    PM = priority_matrix(priorities) * (mappings == mappings.T)

    # the successors' matrix maps, for each task (row), which task is its successor (column)
    # this is a 2D matrix
    S = successor_matrix(successors)

    # initialize response times
    # 3D column vector, each plane for each scenario
    Rmax = np.zeros((s, t, 1), dtype=np.float32)
    Rprev = np.full_like(Rmax, -1.)  # initialized to -1 to bootstrap the main loop

    while not np.allclose(Rmax, Rprev):
        Rprev = Rmax

        # update jitter matrix with current response times
        # 3D column vector, with jitter for each scenario (plane)
        J = jitter_matrix(S, Rmax)

        # initial activation index
        p = 1

        # p-limit mask. when a task reaches its p-limit, its bit is False here.
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

                # WARNING: I probably need some mask to indicate which tasks have already reached their p-limit, and
                # not try to find their W's and R's
                # set to zero all those tasks that have already reached their p-limit
                W = W*pmask

            # once W converges, calculate the response times for this p
            R = W-(p-1)*periods+J

            # do not calculate response times for tasks that have already reached their p-limit
            R = R*pmask

            # update worst-case response times
            Rmax = np.maximum(R, Rmax)

            # stop the p iterations if all W meet the stopping criteria
            # WARNING: not all tasks will reach the same p. I probably need to add a mask to indicate which
            # tasks have already reached their p-limit
            # TODO: I have to update the pmask here, and stop when pmask is all zeroes
            pmask = pmask * np.logical_not(W < STOP)
            if not pmask.any():
                break

            # if no stopping criteria, try with next p
            p += 1

    return Rmax


if __name__ == '__main__':
    wcets = np.array([5, 2, 20, 5, 10, 10], dtype=np.float32).reshape((-1, 1))
    periods = np.array([30, 30, 30, 40, 40, 40], dtype=np.float32).reshape((-1, 1))
    successors = np.array([2, 3, -1, 5, 6, -1], dtype=np.int32).reshape((-1, 1))
    mappings = np.array([1, 3, 2, 2, 3, 1], dtype=np.int32).reshape((-1, 1))
    priorities = np.array([10, 1, 1, 10, 10, 1, 10, 1, 10, 1, 10, 1], dtype=np.int32).reshape((-1, 2), order='F')

    R = analysis(wcets, periods, successors, mappings, priorities)
    print(R)
    # r = np.array([1, 2, 3, 4, 5, 6]).reshape((-1, 1))
    # s = np.array([2, 3, -1, 5, 6, -1]).reshape(-1, 1)
    # S = successor_matrix(s)
    #
    # print(jitter_matrix(S, r))
