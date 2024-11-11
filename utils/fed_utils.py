"""
This file contains the utility functions that can be used
by any training method for the federated learning process.
"""

import copy
import torch
from torch.utils.data import DataLoader
import torch.nn as nn

def local_train(
        clients, 
        index, 
        model_g, 
        task_id, 
        model_old, 
        ep_g, 
        old_client):
    """
    Summary: local training for GLFC

    in detail:
    - copy the global model to the local model
    - update the local model before training
    - train the local model
    - get the local model and the proto gradient


    Args:
        clients: List of client trainers
        index: Index of current client
        model_g: Global model
        task_id: Current task ID
        model_old: List of old models
        ep_g: Global epoch
        old_client: List of old client indices
    """
    # First copy global model to local
    clients[index].model = copy.deepcopy(model_g)
    
    # Update model architecture if needed
    out_features = clients[index].task_size * (task_id + 1)
    if clients[index].model.fc.out_features != out_features:
        in_features = clients[index].model.fc.in_features
        clients[index].model.fc = nn.Linear(in_features, out_features).to(clients[index].device)

    if index in old_client:
        print(f"\nClient {index} (OLD CLIENT)")
        clients[index].beforeTrain(task_id, 0)
    else:
        print(f"\nClient {index} (NEW CLIENT)")
        clients[index].beforeTrain(task_id, 1)

    clients[index].log_class_assignment(index)
    clients[index].update_new_set()
    print(f'entropy signal: {clients[index].signal}')
    
    clients[index].train(ep_g, model_old)
    local_model = clients[index].model.state_dict()
    proto_grad = clients[index].proto_grad_sharing()

    print('*' * 60)
    return local_model, proto_grad

def participant_exemplar_storing(clients, num, model_g, old_client, task_id, clients_index):
    for index in range(num):
        clients[index].model = copy.deepcopy(model_g)
        if index not in clients_index:
            if index in old_client:
                clients[index].beforeTrain(task_id, 0)
            else:
                clients[index].beforeTrain(task_id, 1)
            clients[index].update_new_set()




def model_global_eval(model_g, test_dataset, task_id, task_size, device):
    # model_to_device(model_g, False, device)
    model_g.eval()
    test_dataset.getTestData([0, task_size * (task_id + 1)])
    test_loader = DataLoader(dataset=test_dataset, shuffle=True, batch_size=128)
    correct, total = 0, 0
    for setp, (indexs, imgs, labels) in enumerate(test_loader):
        imgs, labels = imgs.cuda(device), labels.cuda(device)
        with torch.no_grad():
            outputs = model_g(imgs)
        predicts = torch.max(outputs, dim=1)[1]
        correct += (predicts.cpu() == labels.cpu()).sum()
        total += len(labels)
    accuracy = 100 * correct / total
    model_g.train()
    return accuracy