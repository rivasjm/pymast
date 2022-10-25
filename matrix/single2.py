import numpy as np
import torch


def analysis(wcets, periods, priorities):
    r = torch.clone(wcets)
    r_prev = torch.zeros(wcets.shape)
    diag = torch.eye(wcets.shape[0])

    while not torch.allclose(r, r_prev):
        r_prev = r
        r = torch.ceil((r * ((priorities < priorities.t()) / periods.t()) + diag)) @ wcets
    return r


if __name__ == '__main__':
    wcets = torch.tensor([[0.5], [0.5], [3]], requires_grad=True)
    priorities = torch.tensor([[3.0], [2.0], [1.0]], requires_grad=True)
    periods = torch.tensor([[2.0], [3.0], [6.0]], requires_grad=True)

    r = analysis(wcets, periods, priorities)

    s = torch.sum(r)
    s.backward()

    print(r)
