'''
Factory for loading the models
'''

from model.LeNet import LeNet
from model.resnet import resnet18_cbam

def load_model(model_name):
    if model_name == 'LeNet':
        return LeNet()
    elif model_name == 'ResNet18':
        return resnet18_cbam()
    else:
        raise ValueError(f'Model {model_name} not found')