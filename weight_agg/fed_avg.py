import copy
import torch

def fed_avg(models):
    w_avg = copy.deepcopy(models[0])
    for k in w_avg.keys():
        for i in range(1, len(models)):
            w_avg[k] += models[i][k]
        w_avg[k] = torch.div(w_avg[k], len(models))
    return w_avg