from typing import Optional, Union, Tuple
import torch
import torch.nn as nn

class ClassificationModel(nn.Module):
    """Base model for feature extraction with fc layer"""
    
    def __init__(
        self,
        base_model: nn.Module,
        num_classes: int,
        feature_dim: Optional[int] = None
    ):
        super().__init__()
        self.feature_dim = feature_dim or 512
        
        # Setup feature extractor
        self.feature_extractor = self._prepare_full_model(base_model)
        
        # Add classification layer
        self.fc = nn.Linear(self.feature_dim, num_classes)
    
    def _prepare_full_model(self, model: nn.Module) -> nn.Module:
        """Prepares model by preserving fc layer for feature extraction"""
        # If model already has fc layer, preserve it
        if hasattr(model, "fc"):
            if isinstance(model.fc, nn.Sequential):
                self.feature_dim = model.fc[0].in_features
            elif isinstance(model.fc, nn.Linear):
                self.feature_dim = model.fc.in_features
            return model
            
        # If model has classifier, convert to fc
        if hasattr(model, "classifier"):
            if isinstance(model.classifier, nn.Sequential):
                self.feature_dim = model.classifier[-1].in_features
            else:
                self.feature_dim = model.classifier.in_features
            model.fc = model.classifier
            delattr(model, "classifier")
            return model
            
        raise ValueError("Model must have 'fc' or 'classifier' attribute")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass that returns class logits
        Args:
            x: Input tensor of shape (batch_size, channels, height, width)
        Returns:
            Class logits of shape (batch_size, num_classes)
        """
        # Get features
        features = self.feature_extractor(x)
        if isinstance(features, tuple):
            features = features[0]
            
        # Ensure features are flattened
        if len(features.shape) > 2:
            features = features.view(features.size(0), -1)
            
        # Pass through classification layer
        logits = self.fc(features)  # Shape: (batch_size, num_classes)
        return logits

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features without classification"""
        features = self.feature_extractor(x)
        if isinstance(features, tuple):
            features = features[0]
        return features

    def get_params_by_layer(self) -> dict:
        """
        Get model parameters grouped by layer type.
        Useful for federated learning with different learning rates.
        
        Returns:
            Dictionary of parameter groups
        """
        return {
            "feature_extractor": self.feature_extractor.parameters(),
            "classifier": self.fc.parameters()
        }

    def load_state_dict_from_feature_extractor(
        self, 
        state_dict: dict, 
        strict: bool = False
    ) -> None:
        """
        Load state dict for feature extractor only.
        
        Args:
            state_dict: State dict to load
            strict: Whether to strictly enforce that the keys match
        """
        self.feature_extractor.load_state_dict(state_dict, strict=strict)

    @property
    def num_parameters(self) -> int:
        """
        Get total number of trainable parameters.
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad) 