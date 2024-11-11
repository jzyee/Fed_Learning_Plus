'''
Proxy server for GLFC
'''

import torch.nn as nn
import torch
import copy
from torchvision import transforms
from torch.autograd import Variable
import numpy as np
from torch.nn import functional as F
from PIL import Image
import matplotlib.pyplot as plt
import torch.optim as optim
from training.glfc.glfc_model import GLFCModel
from torch.utils.data import DataLoader
import random
# from Fed_utils import *
from training.glfc.glfc_proxy_data import *

class ProxyServer:
    def __init__(self, 
                 device, 
                 learning_rate, 
                 numclass, 
                 feature_extractor, 
                 encode_model, 
                 test_transform,
                 feature_dim):
        super(ProxyServer, self).__init__()
        self.Iteration = 250
        self.learning_rate = learning_rate
        self.model = GLFCModel(
            feature_extractor,
            numclass,
            feature_dim=feature_dim,
            # is_feature_extractor=True
            ).to(device)
        self.encode_model = encode_model.to(device)
        self.monitor_dataset = Proxy_Data(test_transform)
        self.new_set = []
        self.new_set_label = []
        self.numclass = 0
        self.device = device
        self.num_image = 20
        self.pool_grad = None
        self.best_model_1 = None
        self.best_model_2 = None
        self.best_perf = 0

    def dataloader(self, pool_grad):
        '''
        this function is used to create dataloader 
        for monitoring model

        how it works:
        - if pool_grad is not empty, reconstruct the exemplar set
        - if pool_grad is empty, it means the exemplar set is already reconstructed
        - load reconstructed exemplar set into monitor_dataset
        - create dataloader for monitoring model

        args:
        - pool_grad: the gradient of the exemplar set

        why would the examplar set already be reconstructed?
        - because the exemplar set is reconstructed in the beginning of each round

        '''
        self.pool_grad = pool_grad
        self.monitor_loader = None  # Initialize as None
        
        if len(pool_grad) != 0:
            # Reconstruct the exemplar set
            self.reconstruction()
            # load reconstructed exemplar set into monitor_dataset
            self.monitor_dataset.getTestData(self.new_set, self.new_set_label)
            self.monitor_loader = DataLoader(
                dataset=self.monitor_dataset, 
                shuffle=True, 
                batch_size=64, 
                drop_last=True
            )
            self.last_perf = 0
            self.best_model_1 = self.best_model_2

        cur_perf = self.monitor()
        print(f'current performance: {cur_perf}')
        if cur_perf >= self.best_perf:
            self.best_perf = cur_perf
            self.best_model_2 = copy.deepcopy(self.model)

    def model_back(self):
        return [self.best_model_1, self.best_model_2]

    def monitor(self):
        """Monitor performance of the model"""
        self.model.eval()
        
        # If no monitor_loader exists, return 0
        if self.monitor_loader is None:
            return 0
        
        correct, total = 0, 0
        for step, (imgs, labels) in enumerate(self.monitor_loader):
            imgs, labels = imgs.cuda(self.device), labels.cuda(self.device)
            with torch.no_grad():
                outputs = self.model(imgs)
            predicts = torch.max(outputs, dim=1)[1]
            correct += (predicts.cpu() == labels.cpu()).sum()
            total += len(labels)
        
        accuracy = 100 * correct / total if total > 0 else 0
        return accuracy

    def gradient2label(self):
        pool_label = []
        for w_single in self.pool_grad:
            pred = torch.argmin(torch.sum(w_single[-2], dim=-1), dim=-1).detach().reshape((1,)).requires_grad_(False)
            pool_label.append(pred.item())

        return pool_label

    def reconstruction(self):
        '''
        this function is used to reconstruct the exemplar set

        how it works:
        for each class, find the gradient difference between the current model and the exemplar set
        then use the gradient difference to reconstruct the exemplar set

        what we get at the end of this function:
        self.new_set: the reconstructed exemplar set
        self.new_set_label: the label of the reconstructed exemplar set

        
        
        '''
        self.new_set, self.new_set_label = [], []

        tt = transforms.Compose([transforms.ToTensor()])
        tp = transforms.Compose([transforms.ToPILImage()])
        pool_label = self.gradient2label()
        pool_label = np.array(pool_label)
        # print(pool_label)
        class_ratio = np.zeros((1, 100))

        for i in pool_label:
            class_ratio[0, i] += 1

        for label_i in range(100):
            if class_ratio[0, label_i] > 0:
                num_augmentation = self.num_image
                augmentation = []
                
                grad_index = np.where(pool_label == label_i)
                for j in range(len(grad_index[0])):
                    # print('reconstruct_{}, {}-th'.format(label_i, j))
                    grad_truth_temp = self.pool_grad[grad_index[0][j]]

                    dummy_data = torch.randn((1, 3, 32, 32)).to(self.device).requires_grad_(True)
                    label_pred = torch.Tensor([label_i]).long().to(self.device).requires_grad_(False)

                    optimizer = torch.optim.LBFGS([dummy_data, ], lr=0.1)
                    criterion = nn.CrossEntropyLoss().to(self.device)

                    recon_model = copy.deepcopy(self.encode_model)
                    recon_model = recon_model.to(self.device)

                    for iters in range(self.Iteration):
                        def closure():
                            optimizer.zero_grad()
                            pred = recon_model(dummy_data)
                            dummy_loss = criterion(pred, label_pred)

                            dummy_dy_dx = torch.autograd.grad(dummy_loss, recon_model.parameters(), create_graph=True)

                            grad_diff = 0
                            for gx, gy in zip(dummy_dy_dx, grad_truth_temp):
                                grad_diff += ((gx - gy) ** 2).sum()
                            grad_diff.backward()
                            return grad_diff

                        optimizer.step(closure)
                        current_loss = closure().item()

                        if iters == self.Iteration - 1:
                            print(f"for class: {label_i}, current loss: {current_loss}")

                        if iters >= self.Iteration - self.num_image:
                            dummy_data_temp = np.asarray(tp(dummy_data.clone().squeeze(0).cpu()))
                            augmentation.append(dummy_data_temp)

                self.new_set.append(augmentation)
                self.new_set_label.append(label_i)


    