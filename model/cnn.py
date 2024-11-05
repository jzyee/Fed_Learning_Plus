"""
CNN model implementation for Federated Class Incremental Learning
This model serves as the backbone network for feature extraction 
and classification
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List

class CNN(nn.Module):
    """
    Convolutional Neural Network architecture for GLFC
    
    Attributes:
        num_classes (int): Number of output classes
        feature_dim (int): Dimension of feature vectors
        features (nn.Sequential): Feature extraction layers
        classifier (nn.Linear): Final classification layer
    """
    
    def __init__(self, num_classes: int = 10, feature_dim: int = 512) -> None:
        """
        Initialize CNN model
        
        Args:
            num_classes: Number of classes to classify
            feature_dim: Dimension of extracted feature vectors
        """
        super(CNN, self).__init__()
        
        self.num_classes = num_classes
        self.feature_dim = feature_dim
        
        # Feature extraction layers
        self.features = nn.Sequential(
            # Conv block 1
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Conv block 2
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Conv block 3
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Flatten features
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )
        
        # Classification layer
        self.classifier = nn.Linear(256, num_classes)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network
        
        Args:
            x: Input tensor of shape (batch_size, channels, height, width)
            
        Returns:
            Classification logits of shape (batch_size, num_classes)
        """
        features = self.features(x)
        outputs = self.classifier(features)
        return outputs
    
    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract features from input without classification
        
        Args:
            x: Input tensor of shape (batch_size, channels, height, width)
            
        Returns:
            Feature vectors of shape (batch_size, feature_dim)
        """
        return self.features(x)
    
    def get_params(self) -> List[torch.Tensor]:
        """
        Get all model parameters
        
        Returns:
            List of all parameter tensors
        """
        return list(self.parameters())
