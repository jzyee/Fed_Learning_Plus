from typing import Optional
import torch
import torch.nn as nn

class ClassificationModel(nn.Module):
    """Base model for classification with feature extraction capabilities"""
    
    def __init__(
        self,
        base_model: nn.Module,
        num_classes: int,
        feature_dim: Optional[int] = None
    ):
        super().__init__()
        self.feature_dim = feature_dim or 512
        
        # Setup feature extractor and create classification layer
        self.feature_extractor = self._prepare_full_model(base_model)
        self.fc = nn.Linear(self.feature_dim, num_classes)
    
    def _prepare_full_model(self, model: nn.Module) -> nn.Module:
        """
        Prepares model by standardizing on fc layer for feature extraction.
        If model already has feature extraction setup, preserve it.
        """
        # If model already has feature extraction setup (no fc/classifier)
        if not hasattr(model, "fc") and not hasattr(model, "classifier"):
            # Add fc layer for standardization
            model.fc = nn.Identity()
            return model
            
        # Handle models with existing fc/classifier
        if hasattr(model, "fc"):
            if isinstance(model.fc, nn.Sequential):
                self.feature_dim = model.fc[0].in_features
            elif isinstance(model.fc, nn.Linear):
                self.feature_dim = model.fc.in_features
            model.fc = nn.Identity()
        elif hasattr(model, "classifier"):
            if isinstance(model.classifier, nn.Sequential):
                self.feature_dim = model.classifier[-1].in_features
            else:
                self.feature_dim = model.classifier.in_features
            delattr(model, "classifier")
            model.fc = nn.Identity()
            
        return model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass:
        1. Get features through feature_extractor (ends with model.fc = Identity)
        2. Pass through classification layer (self.fc)
        """
        features = self.feature_extractor(x)
        if isinstance(features, tuple):
            features = features[0]
        return self.fc(features)  # Use our classification layer

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features without classification"""
        features = self.feature_extractor(x)
        if isinstance(features, tuple):
            features = features[0]
        return features