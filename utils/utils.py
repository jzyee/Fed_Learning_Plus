import random
import torch
import numpy as np
from transformers import HfArgumentParser
from dataclasses import fields

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



def print_dataclass_help_from_parser(parser: HfArgumentParser):
    '''
    Print the help message for the dataclass
    '''
    # print(dataclass_obj.model_dump_json(indent=2))

    
    print("Help Information for All Arguments:")
    for dataclass_type in parser.dataclass_types:
        print(f"\n{dataclass_type.__name__}:")
        for field in dataclass_type.__dataclass_fields__.values():
            help_msg = field.metadata.get("help", "No help available.")
            print(f"  - {field.name} ({field.type.__name__}): {help_msg} (Default: {field.default})")



def print_help_for_dataclass(dataclass_type):
    print(f"Help for {dataclass_type.__name__}:")
    for field in fields(dataclass_type):
        help_msg = field.metadata.get("help", "No help available.")
        print(f"  - {field.name} ({field.type.__name__}): {help_msg} (Default: {field.default})")