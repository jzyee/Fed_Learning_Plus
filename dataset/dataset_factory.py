from dataset.icifar100 import iCIFAR100
from dataset.icifar10 import iCIFAR10
def load_dataset(dataset_args, train_transform=None, test_transform=None):
    '''
    Get the dataset based on the dataset arguments
    '''

    if dataset_args.dataset_name == 'icifar100':
        train_dataset = iCIFAR100(root='dataset', transform=train_transform, download=True)
        test_dataset = iCIFAR100(root='dataset', test_transform=test_transform, train=False, download=True)
        
    elif dataset_args.dataset_name == 'icifar10':
        train_dataset = iCIFAR10(root='dataset', transform=train_transform, download=True)
        test_dataset = iCIFAR10(root='dataset', test_transform=test_transform, train=False, download=True)
    else:
        raise ValueError(f"Dataset {dataset_args.dataset_name} not supported")
    
    return train_dataset, test_dataset