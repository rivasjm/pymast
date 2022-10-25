import numpy as np

# c = np.array([[0.5, 0.5, 3]])
# o = np.ones((3, 1))
# t = np.array([[0, 0.5, 0.5], [0, 0, 1/3], [0, 0, 0]])
# d = np.eye(3)
# r = np.zeros((1, 3))
#
# # x = o@r*t+d
# r = c@np.ceil(o@r*t+d)
# print(r)
#
# # x = o@r*t+d
# r = c@np.ceil(o@r*t+d)
# print(r)
#
# # x = o@r*t+d
# r = c@np.ceil(o@r*t+d)
# print(r)
#
# # x = o@r*t+d
# r = c@np.ceil(o@r*t+d)
# print(r)
#


def analysis(wcet, periods, priorities):
    vo = np.ones(wcet.shape).transpose()
    ho = np.ones(wcet.shape)
    diag = np.eye(wcet.shape[1])

    # prepare pmatrix
    pm1 = priorities.transpose() @ ho
    pm2 = vo @ priorities
    pm = np.clip(pm1-pm2, 0, 1)

    # prepare imatrix
    im = pm / (periods.transpose() @ ho)

    # initialize response times
    r_prev = np.zeros(wcet.shape)
    r = np.copy(wcet)

    while not np.allclose(r, r_prev):
        r_prev = r
        r = wcet@np.ceil(vo@r*im+diag)
        print(r)

    return r


# def pmatrix(priorities):
#     i1 = priorities.transpose() @ np.ones(priorities.shape)
#     i2 = np.ones(priorities.shape).transpose() @ priorities
#     return np.clip(i1-i2, 0, 1)
#
#
# def imatrix(pmatrix, periods):
#     return pmatrix / (periods.transpose() @ np.ones(periods.shape))


if __name__ == '__main__':
    wcets = np.array([[0.5, 0.5, 3]])
    priorities = np.array([[3, 2, 1]])
    periods = np.array([[2, 3, 6]])

    r = analysis(wcets, periods, priorities)
    print(r)