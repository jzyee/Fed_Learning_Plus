from training.glfc.glfc_model import GLFCModel
import torch
import torch.nn as nn
from typing import Optional

class FAGLFCModel(GLFCModel):
    """
    Fairness-Aware GLFC Model - extends GLFCModel with fairness considerations
    """
    def __init__(
        self,
        base_model: nn.Module,
        num_classes: int,
        feature_dim: Optional[int] = None,
        memory_size: int = 2000
    ):
        super().__init__(
            base_model=base_model,
            num_classes=num_classes,
            feature_dim=feature_dim,
            memory_size=memory_size
        )

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract feature vectors from input"""
        return super().get_features(x)

    def extract_vector(self, x: torch.Tensor) -> torch.Tensor:
        """Alias for get_features to maintain compatibility"""
        return self.get_features(x) 