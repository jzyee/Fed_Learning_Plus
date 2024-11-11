import torch.nn as nn

class LeNet(nn.Module):
    '''
    LeNet model for CIFAR-10 dataset
    '''
    def __init__(self, channel=3, hiden=768, num_classes=10):
        super(LeNet, self).__init__()
        act = nn.Sigmoid
        self.body = nn.Sequential(
            nn.Conv2d(channel, 12, kernel_size=5, padding=5 // 2, stride=2),
            act(),
            nn.Conv2d(12, 12, kernel_size=5, padding=5 // 2, stride=2),
            act(),
            nn.Conv2d(12, 12, kernel_size=5, padding=5 // 2, stride=1),
            act(),
        )
        self.fc = nn.Sequential(
            nn.Linear(hiden, num_classes)
        )

    def forward(self, x):
        '''
        Forward pass of the model
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channel, height, width)
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, num_classes)

        why flaten the output?
        - because the fully connected layer expects a 1D input
        '''
        # pass the input through the body of the model
        out = self.body(x)
        # flatten the output
        out = out.view(out.size(0), -1)
        # pass the output through the fully connected layer
        out = self.fc(out)
        return out