'''
This is the implementation of GLFC.
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import copy
import torchvision.transforms as transforms
from PIL import Image

def entropy(input_):
    '''
    This function is used to calculate the entropy of the input.
    It does the following things:
    - calculate the entropy of the input
    - sum the entropy of each sample

    Input:
    - input_: the input tensor

    Return:
    - entropy: the entropy of the input
    '''
    bs = input_.size(0)
    entropy = -input_ * torch.log(input_ + 1e-5)
    entropy = torch.sum(entropy, dim=1)
    return entropy

def entropy_signal(self, loader):
    '''
    This function is used to get the entropy signal.
    It does the following things:
    - evaluate the entropy of the current model
    - determine whether to update the new set
    - update the last entropy

    Input:
    - loader: the dataloader of the train data

    Return:
    - res: whether to update the new set
    '''
    self.model.eval()
    start_ent = True
    res = False

    for step, (indexs, imgs, labels) in enumerate(loader):
        imgs, labels = imgs.cuda(self.device), labels.cuda(self.device)
        with torch.no_grad():
            outputs = self.model(imgs)
        softmax_out = nn.Softmax(dim=1)(outputs)
        ent = entropy(softmax_out)

        if start_ent:
            all_ent = ent.float().cpu()
            all_label = labels.long().cpu()
            start_ent = False
        else:
            all_ent = torch.cat((all_ent, ent.float().cpu()), 0)
            all_label = torch.cat((all_label, labels.long().cpu()), 0)

    overall_avg = torch.mean(all_ent).item()
    print(overall_avg)
    if overall_avg - self.last_entropy > 1.2:
        res = True
    
    self.last_entropy = overall_avg

    self.model.train()

    return res

def proto_grad_sharing(self):
    '''
    This function is used to get the prototype gradient sharing.
    It does the following things:
    - get the prototype gradient

    Input:
    - None

    Return:
    - proto_grad: the prototype gradient
    '''
    if self.signal:
        proto_grad = self.prototype_mask()
    else:
        proto_grad = None

    return proto_grad

def prototype_mask(self):
    '''
    This function is used to get the prototype mask.
    It does the following things:
    - get the prototype
    - get the prototype gradient

    Input:
    - None

    Return:
    - proto: the prototype
    - proto_grad: the prototype gradient
    '''
    tt = transforms.Compose([transforms.ToTensor()])
    tp = transforms.Compose([transforms.ToPILImage()])
    iters = 50
    criterion = nn.CrossEntropyLoss().to(self.device)
    proto = []
    proto_grad = []

    for i in self.current_class:
        images = self.train_dataset.get_image_class(i)
        class_mean, feature_extractor_output = self.compute_class_mean(images, self.transform)
        dis = class_mean - feature_extractor_output
        dis = np.linalg.norm(dis, axis=1)
        pro_index = np.argmin(dis)
        proto.append(images[pro_index])

    for i in range(len(proto)):
        self.model.eval()
        data = proto[i]
        label = self.current_class[i]
        data = Image.fromarray(data)
        label_np = label
        
        data, label = tt(data), torch.Tensor([label]).long()
        data, label = data.cuda(self.device), label.cuda(self.device)
        data = data.unsqueeze(0).requires_grad_(True)
        target = get_one_hot(label, self.numclass, self.device)

        opt = optim.SGD([data, ], lr=self.learning_rate / 10, weight_decay=0.00001)
        proto_model = copy.deepcopy(self.model)
        proto_model = model_to_device(proto_model, False, self.device)

        for ep in range(iters):
            outputs = proto_model(data)
            loss_cls = F.binary_cross_entropy_with_logits(outputs, target)
            opt.zero_grad()
            loss_cls.backward()
            opt.step()

        self.encode_model = model_to_device(self.encode_model, False, self.device)
        data = data.detach().clone().to(self.device).requires_grad_(False)
        outputs = self.encode_model(data)
        loss_cls = criterion(outputs, label)
        dy_dx = torch.autograd.grad(loss_cls, self.encode_model.parameters())
        original_dy_dx = list((_.detach().clone() for _ in dy_dx))
        proto_grad.append(original_dy_dx)

    return proto_grad