from typing import Tuple, Any
from torchvision import datasets, transforms
from dataset.tumor_dataset import MRI_Tumor_17
from dataset.icifar10 import iCIFAR10
from dataset.icifar100 import iCIFAR100
from dataclasses import dataclass
def load_dataset(
    dataset_args: dataclass,
    train_transform: transforms.Compose,
    test_transform: transforms.Compose
) -> Tuple[Any, Any]:
    """
    Load and return train and test datasets
    """
    if dataset_args.dataset_name.lower() == "mri_tumor_17":
        # For tumor dataset, we only create one instance and split it internally
        train_dataset = MRI_Tumor_17(
            root='dataset',
            train=True,  # This will be used to determine which transform to apply
            transform=train_transform,
            download=True
        )
        test_dataset = MRI_Tumor_17(
            root='dataset',
            train=False,
            test_transform=test_transform,
            download=True
        )
        return train_dataset, test_dataset
    
    # For CIFAR datasets, create separate train and test instances
    if dataset_args.dataset_name.lower() == "icifar10":
        train_dataset = iCIFAR10(
            root='dataset',
            train=True,
            transform=train_transform,
            download=True
        )
        test_dataset = iCIFAR10(
            root='dataset',
            train=False,
            test_transform=test_transform,
            download=True
        )
    elif dataset_args.dataset_name.lower() == "icifar100":
        train_dataset = iCIFAR100(
            root='dataset',
            train=True,
            transform=train_transform,
            download=True
        )
        test_dataset = iCIFAR100(
            root='dataset',
            train=False,
            test_transform=test_transform,
            download=True
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset_args.dataset_name}")
    
    return train_dataset, test_dataset