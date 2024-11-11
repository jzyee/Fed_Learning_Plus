import random
import torch
import numpy as np

def setup_seed(seed):
    '''
    Set the seed for the random number generators in PyTorch, NumPy, and Python
    '''
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def weights_init(m):
    try:
        if hasattr(m, "weight"):
            m.weight.data.uniform_(-0.5, 0.5)
    except Exception:
        print('warning: failed in weights_init for %s.weight' % m._get_name())
    try:
        if hasattr(m, "bias"):
            m.bias.data.uniform_(-0.5, 0.5)
    except Exception:
        print('warning: failed in weights_init for %s.bias' % m._get_name())

def get_one_hot(target, num_classes, device):
    '''
    Get one-hot encoding of target(labels)
    '''
    one_hot=torch.zeros(target.shape[0], num_classes).cuda(device)
    one_hot=one_hot.scatter(dim=1,index=target.long().view(-1,1),value=1.)
    return one_hot