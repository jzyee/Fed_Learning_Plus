#!/usr/bin/env python3
import sys
import os
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataclasses import dataclass, field
from transformers import HfArgumentParser
import torch
import os.path as osp
from typing import Dict, List, Optional

from training.icarl.icarl_fl import iCaRLPlusFL
from model.resnet import ResNet18
from utils.data_manager import DataManager
from utils.setup import setup_seed, setup_logger
from utils.img_transforms import train_transforms, test_transforms

@dataclass
class ModelArgs:
    model_name: str = field(
        default="ResNet18",
        metadata={"help": "model to use for training"})

@dataclass
class DatasetArgs:
    dataset_name: str = field(
        default="cifar10",
        metadata={"help": "dataset to use for training"})
    img_size: int = field(
        default=32,
        metadata={"help": "image size"})

@dataclass
class TrainingArgs:
    batch_size: int = field(
        default=128,
        metadata={"help": "batch size"})
    seed: int = field(
        default=42,
        metadata={"help": "random seed for training"})
    num_clients: int = field(
        default=5,
        metadata={"help": "number of federated clients"})
    local_clients: int = field(
        default=3,
        metadata={"help": "number of local clients per round"})
    num_classes: int = field(
        default=10,
        metadata={"help": "total number of classes"})
    device: str = field(
        default="cuda",
        metadata={"help": "device to use for training"})
    memory_size: int = field(
        default=2000,
        metadata={"help": "size of exemplar memory per client"})
    epochs_local: int = field(
        default=5,
        metadata={"help": "number of local epochs"})
    epochs_global: int = field(
        default=100,
        metadata={"help": "number of federation rounds"})
    learning_rate: float = field(
        default=0.1,
        metadata={"help": "learning rate"})
    task_size: int = field(
        default=2,
        metadata={"help": "number of classes per task"})
    tasks_global: int = field(
        default=5,
        metadata={"help": "total number of tasks"})
    output_dir: str = field(
        default="./output",
        metadata={"help": "output directory"})

def main() -> None:
    """
    Main training function for iCaRL with Federated Learning
    """
    # Initialize argument parser
    parser = HfArgumentParser((ModelArgs, DatasetArgs, TrainingArgs))
    model_args, dataset_args, training_args = parser.parse_args_into_dataclasses()

    # Setup output directory
    output_dir = osp.join(
        training_args.output_dir,
        f"{model_args.model_name}_{dataset_args.dataset_name}_seed{training_args.seed}"
    )
    os.makedirs(output_dir, exist_ok=True)

    # Initialize logger
    out_file = open(osp.join(output_dir, f'log_tar_{training_args.task_size}.txt'), 'w')
    log_str = f'method_iCaRLFL, task_size_{training_args.task_size}, learning_rate_{training_args.learning_rate}'
    out_file.write(log_str + '\n')
    out_file.flush()

    # Set random seed
    setup_seed(training_args.seed)

    # Setup device
    device = torch.device(training_args.device)

    # Initialize transforms
    train_transform = train_transforms(dataset_args.img_size)
    test_transform = test_transforms(dataset_args.img_size)

    # Load dataset
    train_dataset, test_dataset = load_dataset(dataset_args, train_transform, test_transform)

    # Initialize model
    feature_extractor = ResNet18(num_classes=training_args.num_classes).to(device)

    # Initialize iCaRL+FL
    icarl_fl = iCaRLPlusFL(
        device=device,
        num_classes=training_args.num_classes,
        memory_size=training_args.memory_size,
        num_clients=training_args.num_clients
    )
    icarl_fl.global_model = feature_extractor

    # Calculate epochs per task
    epochs_per_task = training_args.epochs_global // training_args.tasks_global
    print(f"\nTraining Configuration:")
    print(f"Total epochs: {training_args.epochs_global}")
    print(f"Total tasks: {training_args.tasks_global}")
    print(f"Epochs per task: {epochs_per_task}")
    print(f"Classes per task: {training_args.task_size}\n")

    old_task_id = -1
    classes_learned = training_args.task_size

    for epoch_g in range(training_args.epochs_global):
        task_id = epoch_g // epochs_per_task
        print(f"\nGlobal epoch {epoch_g + 1}/{training_args.epochs_global}")
        print(f"Current task: {task_id + 1}/{training_args.tasks_global}")

        # Handle new task
        if task_id != old_task_id and old_task_id != -1:
            classes_learned += training_args.task_size
            icarl_fl.global_model.incremental_learning(classes_learned)
            icarl_fl.global_model.to(device)

        # Local training
        client_models = {}
        for client_id in range(training_args.local_clients):
            client_models[client_id] = train_client(
                icarl_fl=icarl_fl,
                client_id=client_id,
                train_dataset=train_dataset,
                task_id=task_id,
                training_args=training_args
            )

        # Federated averaging
        icarl_fl.federated_averaging(client_models)

        # Evaluate global model
        acc_global = evaluate_global_model(
            model=icarl_fl.global_model,
            test_dataset=test_dataset,
            task_id=task_id,
            device=device
        )

        # Log results
        log_str = f'Task: {task_id}, Round: {epoch_g} Accuracy = {acc_global:.2f}%'
        out_file.write(log_str + '\n')
        out_file.flush()
        print(f'Classification accuracy of global model at round {epoch_g}: {acc_global:.3f}\n')

        old_task_id = task_id

    # Save final model
    torch.save(
        icarl_fl.global_model.state_dict(),
        osp.join(output_dir, "final_model.pth")
    )

def train_client(
    icarl_fl: iCaRLPlusFL,
    client_id: int,
    train_dataset: torch.utils.data.Dataset,
    task_id: int,
    training_args: TrainingArgs
) -> nn.Module:
    """
    Train a client model
    """
    client_model = deepcopy(icarl_fl.global_model)
    optimizer = torch.optim.SGD(
        client_model.parameters(),
        lr=training_args.learning_rate,
        momentum=0.9,
        weight_decay=5e-4
    )

    for epoch in range(training_args.epochs_local):
        for images, labels in train_dataset:
            images = images.to(training_args.device)
            labels = labels.to(training_args.device)

            optimizer.zero_grad()
            outputs = client_model(images)

            # Get old model outputs for distillation
            with torch.no_grad():
                old_outputs = icarl_fl.global_model(images)

            loss = icarl_fl.knowledge_distillation_loss(
                outputs=outputs,
                targets=labels,
                old_outputs=old_outputs
            )

            loss.backward()
            optimizer.step()

    return client_model

def evaluate_global_model(
    model: nn.Module,
    test_dataset: torch.utils.data.Dataset,
    task_id: int,
    device: torch.device
) -> float:
    """
    Evaluate global model performance
    """
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_dataset:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return 100.0 * correct / total

if __name__ == "__main__":
    main() 