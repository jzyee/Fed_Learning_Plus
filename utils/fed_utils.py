"""
This file contains the utility functions that can be used
by any training method for the federated learning process.
"""

import copy
import torch
from torch.utils.data import DataLoader

def local_train(clients, index, model_g, task_id, model_old, ep_g, old_client):
    """
    Summary: local training for GLFC

    in detail:
    - copy the global model to the local model
    - update the local model before training
    - train the local model
    - get the local model and the proto gradient


    Args:
        clients: the trainers
        index: the index of the client
        model_g: the global model
        task_id: the id of the task
        model_old: the old model
        ep_g: the global epoch
        old_client: the old clients
    """
    clients[index].model = copy.deepcopy(model_g)

    if index in old_client:
        print(f"\nClient {index} (OLD CLIENT)")
        clients[index].beforeTrain(task_id, 0)
    else:
        print(f"\nClient {index} (NEW CLIENT)")
        clients[index].beforeTrain(task_id, 1)

    # for logging of the class assignment
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