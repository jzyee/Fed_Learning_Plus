from typing import Optional, Union
import torch
import torch.nn as nn

class GLFCModel(nn.Module):
    """
    Wrapper for models in GLFC that handles both feature extractors and classifiers
    """
    def __init__(
        self,
        base_model: nn.Module,
        num_classes: int,
        feature_dim: int = 512,
        is_feature_extractor: bool = False
    ):
        """
        Args:
            base_model: Base model (either feature extractor or full classifier)
            num_classes: Number of classes
            feature_dim: Dimension of feature vector
            is_feature_extractor: Whether base_model is only a feature extractor
        """
        super().__init__()
        self.is_feature_extractor = is_feature_extractor
        
        if is_feature_extractor:
            self.feature_extractor = base_model
            self.classifier = nn.Linear(feature_dim, num_classes)
        else:
            # For models that already include classification layer
            self.feature_extractor = base_model
            self.classifier = nn.Identity()
            
            # Remove the last classification layer
            if hasattr(self.feature_extractor, "fc"):
                feature_dim = self.feature_extractor.fc.in_features
                self.feature_extractor.fc = nn.Identity()
            elif hasattr(self.feature_extractor, "classifier"):
                feature_dim = self.feature_extractor.classifier[-1].in_features
                self.feature_extractor.classifier[-1] = nn.Identity()
                
            self.classifier = nn.Linear(feature_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.feature_extractor(x)
        return self.classifier(features)

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.feature_extractor(x) 