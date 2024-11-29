"""
This file contains the utility functions that can be used
by any training method for the federated learning process.
"""

import copy
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from torchvision import transforms
from typing import Optional

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




def model_global_eval(model, test_dataset, task_id, task_size, device):
    # # Add transforms to convert PIL Images to tensors
    # transform = transforms.Compose([
    #     transforms.ToTensor(),
    #     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))  # CIFAR-10 normalization
    # ])
    
    # # Apply transforms to the test dataset if not already applied
    # if not hasattr(test_dataset, 'transform') or test_dataset.transform is None:
    #     test_dataset.transform = transform

    model.to(device)
    model.eval()
    test_dataset.getTestData([0, task_size * (task_id + 1)])

    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
    
    correct, total = 0, 0
    for setp, (indexs, imgs, labels) in enumerate(test_loader):
        imgs, labels = imgs.cuda(device), labels.cuda(device)
        with torch.no_grad():
            outputs = model(imgs)
        predicts = torch.max(outputs, dim=1)[1]
        correct += (predicts.cpu() == labels.cpu()).sum()
        total += len(labels)
    accuracy = 100 * correct / total
    model.train()
    return accuracy

def visualize_class_forgetting(model, test_dataset, task_id, task_size, memory_size, device, output_dir, global_round=None):
    """
    Create and save a heatmap visualization of class-wise performance with unique filename.
    Also maintains a CSV file of historical results.
    """
    import numpy as np
    import seaborn as sns
    import matplotlib.pyplot as plt
    from datetime import datetime
    import os
    import pandas as pd
    
    # Setup CSV file path for storing results
    csv_path = os.path.join(output_dir, "class_forgetting_results.csv")
    
    model.to(device)
    model.eval()
    
    # Calculate total number of classes
    total_classes = task_size * (task_id + 1)
    
    # Initialize arrays for class-wise accuracy
    class_correct = np.zeros(total_classes)
    class_total = np.zeros(total_classes)
    
    # Evaluate each task separately
    for t in range(task_id + 1):
        start_class = t * task_size
        end_class = (t + 1) * task_size
        test_dataset.getTestData([start_class, end_class])
        test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
        
        with torch.no_grad():
            for _, imgs, labels in test_loader:
                imgs, labels = imgs.cuda(device), labels.cuda(device)
                outputs = model(imgs)
                predicts = torch.max(outputs, dim=1)[1]
                
                for label, pred in zip(labels, predicts):
                    label_id = label.item()
                    if start_class <= label_id < end_class:
                        class_total[label_id] += 1
                        if label_id == pred.item():
                            class_correct[label_id] += 1
    
    # Calculate accuracy percentages
    accuracies = np.where(class_total > 0, 
                         (class_correct / class_total) * 100, 
                         0)
    
    # Create or update CSV with results
    round_results = {
        'round': global_round if global_round is not None else 0,
        'task': task_id,
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    # if csv exists and is first task and round, delete the csv file
    if os.path.exists(csv_path) and task_id == 0 and global_round == 0:
        os.remove(csv_path)

    # Add accuracy for each class with explicit class IDs
    for class_id in range(total_classes):
        round_results[f'class_{class_id}'] = accuracies[class_id]
    
    # Read existing results or create new DataFrame
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        # Append new results
        df = pd.concat([df, pd.DataFrame([round_results])], ignore_index=True)
    else:
        df = pd.DataFrame([round_results])
    
    # Save updated results
    df.to_csv(csv_path, index=False)
    
    # Create current state heatmap
    plt.figure(figsize=(12, 8))
    current_heatmap = accuracies.reshape(task_id + 1, task_size)
    sns.heatmap(
        current_heatmap,
        xticklabels=[f"class_{x}" for x in range(total_classes)],
        annot=True,
        fmt='.1f',
        cmap='YlOrRd',
        vmin=0,
        vmax=100,
        cbar_kws={'label': 'Accuracy %'}
    )
    
    # Customize current state plot
    title = f"Class-wise Performance Heatmap"
    if global_round is not None:
        title += f" (Round {global_round})"
    plt.title(title)
    plt.xlabel(f"Classes within Task (Task Size: {task_size})")
    plt.ylabel("Task ID")
    
    # Add task and class labels
    task_labels = [f"Task {i}" for i in range(task_id + 1)]
    class_labels = [f"Class {i}" for i in range(task_size)]
    plt.yticks(np.arange(task_id + 1) + 0.5, task_labels)
    plt.xticks(np.arange(task_size) + 0.5, class_labels)
    
    # Save current state heatmap
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    curr_filename = f"class_forgetting_current_t{task_id}_r{global_round}_m{memory_size}.png"
    curr_save_path = os.path.join(output_dir, curr_filename)
    plt.savefig(curr_save_path, bbox_inches='tight', dpi=300)
    plt.close()
    
    print(f"Current state heatmap saved as: {curr_filename}")
    
    # Create progression heatmap if we have multiple rounds
    if global_round is not None and len(df) > 1:
        plot_forgetting_heatmap(output_dir, task_id, task_size, memory_size, global_round)
    
    model.train()
    return accuracies

def plot_forgetting_heatmap(output_dir: str, task_id: int, task_size: int, memory_size: int, global_round: Optional[int] = None):
    """
    Plot heatmap using stored results from class_forgetting_results.csv
    
    Args:
        output_dir: Directory containing the results CSV
        task_id: Current task ID
        task_size: Number of classes per task
        global_round: Optional current global round number
    """
    import numpy as np
    import seaborn as sns
    import matplotlib.pyplot as plt
    from datetime import datetime
    import os
    import pandas as pd
    
    csv_path = os.path.join(output_dir, "class_forgetting_results.csv")
    if not os.path.exists(csv_path):
        print(f"No results file found at {csv_path}")
        return
        
    # Read stored results
    df = pd.read_csv(csv_path)
    
    # Get data for all rounds up to current task
    task_data = df[df['task'] <= task_id].copy()
    task_data = task_data.sort_values(['round', 'task'])
    
    if len(task_data) == 0:
        print("No data available for plotting")
        return
    
    # Calculate total classes
    total_classes = task_size * (task_id + 1)
    
    plt.figure(figsize=(15, 10))
    
    # Create matrix for heatmap
    rounds = sorted(task_data['round'].unique())
    heatmap_data = []
    
    # Build heatmap data ensuring correct class alignment
    for round_num in rounds:
        round_accuracies = []
        round_data = task_data[task_data['round'] == round_num].iloc[-1]
        
        # Explicitly collect accuracies for each class in order
        for class_id in range(total_classes):
            col_name = f'class_{class_id}'
            accuracy = round_data[col_name] if col_name in round_data else 0.0
            round_accuracies.append(accuracy)
        
        heatmap_data.append(round_accuracies)
    
    heatmap_data = np.array(heatmap_data)
    
    # Create heatmap
    sns.heatmap(
        heatmap_data,
        annot=True,
        fmt='.1f',
        cmap='YlOrRd',
        vmin=0,
        vmax=100,
        cbar_kws={'label': 'Accuracy %'}
    )
    
    # Customize plot
    plt.title(f"Class-wise Performance Progression (Up to Task {task_id})")
    plt.xlabel("Class ID")
    plt.ylabel("Round")
    
    rounds_cpy = copy.deepcopy(rounds)
    rounds_cpy = np.array(rounds_cpy)
    
    # Add round labels
    if len(rounds) < 8:
        round_labels = [f"Round {r}" for r in rounds]
    else:
        num_ticks = 8
        tick_pos = np.linspace(0, len(rounds) - 1, num_ticks, dtype=int)
        round_labels = [f"{r}" for r in rounds_cpy[tick_pos]]
    plt.yticks(np.arange(len(round_labels)) + 0.5, round_labels)
    
    # Add class labels with task boundaries
    class_labels = [f"class_{x}" for x in range(total_classes)]
    plt.xticks(np.arange(len(class_labels)) + 0.5, class_labels, rotation=45, ha='right')
    
    # Add vertical lines to separate tasks
    for t in range(1, task_id + 1):
        plt.axvline(x=t * task_size, color='black', linestyle='--', linewidth=0.5)
    
    # Save heatmap
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"class_forgetting_progression_t{task_id}_r{global_round}_m{memory_size}.png"
    save_path = os.path.join(output_dir, filename)
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()
    
    print(f"Progression heatmap saved as: {filename}")

