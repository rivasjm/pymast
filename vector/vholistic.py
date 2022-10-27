import numpy as np


def successor_matrix(succesors):
    """Builds the successor matrix from a flat successor list. Task id's start from 1"""
    s = np.zeros((succesors.size, succesors.size), dtype=np.int32)
    for i, v in enumerate(succesors):
        if v > -1:
            s[i, v-1] = 1
    return s


def jitter_matrix(smatrix, rmatrix):
    """Builds the jitter matrix. Assumes at most 1 successor per task"""
    return S.T @ r


if __name__ == '__main__':
    r = np.array([1, 2, 3, 4, 5, 6]).reshape((-1, 1))
    s = np.array([2, 3, -1, 5, 6, -1]).reshape(-1, 1)
    S = successor_matrix(s)

    print(jitter_matrix(S, r))
