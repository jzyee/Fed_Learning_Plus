#!/usr/bin/env python3
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataclasses import dataclass, field
from typing import Optional
import torch
import os.path as osp
from transformers import HfArgumentParser
from training.icarl.icarl import iCaRL
from model.model_factory import load_model
from utils.img_transforms import train_transforms, test_transforms
from dataset.dataset_factory import load_dataset

@dataclass
class ModelArgs:
    model_name: str = field(
        default="ResNet18",
        metadata={"help": "model to use for training"})

@dataclass
class EncoderArgs:
    encoder_name: str = field(
        default="LeNet",
        metadata={"help": "encoder to use for training"})

@dataclass
class DatasetArguments:
    """Arguments for dataset configuration"""
    dataset_name: str = field(
        default="cifar10",
        metadata={"help": "Name of the dataset to use"}
    )
    img_size: int = field(
        default=32,
        metadata={"help": "Size of input images"}
    )
    data_path: str = field(
        default="data",
        metadata={"help": "Path to dataset"}
    )
    num_workers: int = field(
        default=4,
        metadata={"help": "Number of workers for data loading"}
    )
    pin_memory: bool = field(
        default=True,
        metadata={"help": "Pin memory for data loading"}
    )

@dataclass
class TrainingArguments:
    """Arguments for iCaRL+FL training"""
    output_dir: str = field(
        default="./output",
        metadata={"help": "Output directory"}
    )
    num_clients: int = field(
        default=30,
        metadata={"help": "Number of clients"}
    )
    num_classes: int = field(
        default=10,
        metadata={"help": "Number of data classes in the first task"}
    )
    task_size: int = field(
        default=10,
        metadata={"help": "Number of classes per task"}
    )
    memory_size: int = field(
        default=2000,
        metadata={"help": "Size of exemplar memory"}
    )
    epochs_local: int = field(
        default=20,
        metadata={"help": "Number of local epochs of each global round"}
    )
    epochs_global: int = field(
        default=100,
        metadata={"help": "Number of global epochs"}
    )
    learning_rate: float = field(
        default=2.0,
        metadata={"help": "Learning rate"}
    )
    batch_size: int = field(
        default=128,
        metadata={"help": "Training batch size"}
    )
    local_clients: int = field(
        default=10,
        metadata={"help": "Number of local clients"}
    )
    tasks_global: int = field(
        default=10,
        metadata={"help": "Total number of tasks"}
    )
    iid_level: int = field(
        default=6,
        metadata={"help": "Number of data classes per local client"}
    )
    device: str = field(
        default="cuda" if torch.cuda.is_available() else "cpu",
        metadata={"help": "Device to use for training"}
    )
    seed: int = field(
        default=42,
        metadata={"help": "Random seed"}
    )


    

def main():
    # Parse arguments
    parser = HfArgumentParser((ModelArgs, DatasetArguments, TrainingArguments))
    model_args, dataset_args, training_args = parser.parse_args_into_dataclasses()
    
    # Set random seed
    torch.manual_seed(training_args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(training_args.seed)
    
    # Create output directory
    os.makedirs(training_args.output_dir, exist_ok=True)
    
    # Define transformations
    train_transform = train_transforms(dataset_args.img_size)
    test_transform = test_transforms(dataset_args.img_size)
    
    # Load dataset using the common load_dataset function
    train_dataset, test_dataset = load_dataset(
        dataset_args,
        train_transform,
        test_transform
    )
    
    # Before creating test_loader
    print(f"Test dataset size before loader creation: {len(test_dataset)}")
    print(f"Test dataset classes: {test_dataset.classes if hasattr(test_dataset, 'classes') else 'No classes found'}")
    print(f"Batch size: {training_args.batch_size}")
    
    # Create test loader
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=training_args.batch_size,
        shuffle=False,
        num_workers=dataset_args.num_workers,
        pin_memory=dataset_args.pin_memory
    )
    
    # After creating test_loader
    print(f"Number of batches in test_loader: {len(test_loader)}")
    # Test first batch
    test_iter = iter(test_loader)
    try:
        first_batch = next(test_iter)
        print(f"First batch shapes - Images: {first_batch[0].shape}, Labels: {first_batch[1].shape}")
    except StopIteration:
        print("ERROR: test_loader is empty!")
    except Exception as e:
        print(f"ERROR loading first batch: {str(e)}")
    
    print("model loading")
    # Initialize feature extractor
    feature_extractor = load_model(model_args.model_name)   
    
    print("iCaRL+FL initializing")
    # Initialize iCaRL+FL
    icarl_fl = iCaRL(training_args, feature_extractor, train_dataset)
    server, clients = icarl_fl.get_components()
    
    print("training loop")
    # Training loop
    num_tasks = training_args.tasks_global // training_args.task_size
    print(f"num_tasks: {num_tasks}")
    for task_id in range(num_tasks):
        print(f"\nTask {task_id + 1}/{num_tasks}")
        
        # Prepare clients for new task
        for client_idx, client in enumerate(clients):
            group = 1 if client_idx < training_args.num_clients // 2 else 0
            client.beforeTrain(task_id, group)
            client.log_class_assignment(client_idx)
        
        # Local training rounds
        for epoch in range(training_args.epochs_local):
            print(f"\nLocal training epoch {epoch + 1}/{training_args.epochs_local}")
            
            # Train each client (feature extractor only)
            client_feature_extractors = {}
            for client_idx, client in enumerate(clients):
                # Get current feature extractor from server
                server_model = server.get_current_model()
                if server_model is None:
                    raise RuntimeError("Failed to get server model for training")
                client.train_feature_extractor(server_model)
                client_feature_extractors[client_idx] = client.get_feature_extractor()
            
            # Aggregate feature extractors only
            server.aggregate_feature_extractors(client_feature_extractors)
            
            # Server updates classifier using global data
            server.update_classifier()
            
            # Evaluate global model
            accuracy = server.evaluate(test_loader)
            print(f"Global model accuracy: {accuracy:.2f}%")
            
            # Save checkpoint
            checkpoint = {
                "task_id": task_id,
                "epoch": epoch,
                "model_state": server.model.state_dict(),
                "accuracy": accuracy
            }
            torch.save(
                checkpoint,
                osp.join(training_args.output_dir, f"checkpoint_task{task_id}_epoch{epoch}.pt")
            )

if __name__ == "__main__":
    main() 