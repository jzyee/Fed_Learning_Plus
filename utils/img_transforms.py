from torchvision import transforms


def train_transforms(img_size):
    '''
    Define the transformations for the training data
    '''
    return transforms.Compose(
        [
            transforms.RandomCrop((img_size, img_size), padding=4),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.24705882352941178),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
        ]
    )

def test_transforms(img_size):
    '''
    Define the transformations for the test data
    ''' 
    return transforms.Compose(
        [
            transforms.Resize((img_size, img_size)), 
            transforms.ToTensor(), 
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
        ]
    )