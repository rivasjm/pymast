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


def analysis(wcets, periods, successors, mappings, priorities):
    assert wcets.shape == periods.shape == successors.shape == mappings.shape == priorities.shape
    assert wcets.shape[1] == 1

    S = successor_matrix(successors)    # successors matrix
    M = mappings == mappings.T          # mappings matrix
    P = priorities < priorities.T       # priorities matrix
    PM = P * M                          # actual priorities' matrix (considering mapping)

    R = np.zeros_like(wcets)
    R_prev = np.full_like(wcets, -1)

    while not np.allclose(R, R_prev):
        R_prev = R
        J = jitter_matrix(S, R)
        R_max = np.zeros_like(wcets)

        p = 1
        STOP = p * periods
        while True:
            W = np.zeros_like(wcets)
            W_prev = np.full_like(wcets, -1)

            # W iteration (convergence of W)
            while not np.allclose(W, W_prev):
                W_prev = W
                W = np.ceil((PM*W+PM*J.T)/periods.T)@wcets + p*wcets
                R = W-(p-1)*periods+J
                R_max = np.maximum(R, R_max)

            if np.less(W, STOP).all():
                break
            p += 1

        R = R_max

    return R


if __name__ == '__main__':
    wcets = np.array([5, 2, 20, 5, 10, 10], dtype=np.float32).reshape((-1, 1))
    periods = np.array([30, 30, 30, 40, 40, 40], dtype=np.float32).reshape((-1, 1))
    successors = np.array([2, 3, -1, 5, 6, -1], dtype=np.int32).reshape((-1, 1))
    mappings = np.array([1, 3, 2, 2, 3, 1], dtype=np.int32).reshape((-1, 1))
    priorities = np.array([10, 1, 1, 10, 10, 1], dtype=np.int32).reshape((-1, 1))

    R = analysis(wcets, periods, successors, mappings, priorities)
    print(R)
    # r = np.array([1, 2, 3, 4, 5, 6]).reshape((-1, 1))
    # s = np.array([2, 3, -1, 5, 6, -1]).reshape(-1, 1)
    # S = successor_matrix(s)
    #
    # print(jitter_matrix(S, r))
