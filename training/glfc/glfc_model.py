from typing import Optional, List
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import numpy as np
from model.classification_model import ClassificationModel

class GLFCModel(ClassificationModel):
    """
    GLFC Model with exemplar management and feature extraction capabilities
    """
    def __init__(
        self,
        base_model: nn.Module,
        num_classes: int,
        feature_dim: Optional[int] = None,
        # is_feature_extractor: bool = False,
        memory_size: int = 2000
    ):
        super().__init__(
            base_model=base_model,
            num_classes=num_classes,
            feature_dim=feature_dim,
            # is_feature_extractor=is_feature_extractor
        )
        self.memory_size = memory_size
        self.exemplar_set: List = []
        self.class_mean_set: List = []
        self.learned_classes: List[int] = []
        self.learned_numclass = 0
    
    def incremental_learning(self, num_classes: int) -> None:
        """Incremental learning"""
        weight = self.fc.weight.data
        bias = self.fc.bias.data
        in_feature = self.fc.in_features
        out_feature = self.fc.out_features

        # update the number of classes
        self.fc = nn.Linear(in_feature, num_classes, bias=True)
        self.fc.weight.data[:out_feature] = weight
        self.fc.bias.data[:out_feature] = bias

    def construct_exemplar_set(
        self,
        images: List,
        m: int,
        transform: Optional[nn.Module] = None
    ) -> None:
        """Construct exemplar set for a class"""
        class_mean, feature_extractor_output = self.compute_class_mean(images, transform)
        exemplar = []
        now_class_mean = np.zeros((1, self.feature_dim))
     
        for i in range(m):
            # Find image whose feature is closest to the mean
            x = class_mean - (now_class_mean + feature_extractor_output) / (i + 1)
            x = np.linalg.norm(x, axis=1)
            index = np.argmin(x)
            now_class_mean += feature_extractor_output[index]
            exemplar.append(images[index])

        self.exemplar_set.append(exemplar)

    def reduce_exemplar_sets(self, m: int) -> None:
        """Reduce exemplar set size"""
        for index in range(len(self.exemplar_set)):
            self.exemplar_set[index] = self.exemplar_set[index][:m]

    def compute_class_mean(
        self,
        images: List,
        transform: Optional[nn.Module] = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute mean feature vector for a class"""
        if transform is None:
            transform = self.default_transform
            
        x = self._transform_images(images, transform)
        features = F.normalize(self.get_features(x)).detach().cpu().numpy()
        class_mean = np.mean(features, axis=0)
        return class_mean, features

    def _transform_images(
        self,
        images: List,
        transform: nn.Module
    ) -> torch.Tensor:
        """Transform list of images to tensor"""
        data = transform(Image.fromarray(images[0])).unsqueeze(0)
        for index in range(1, len(images)):
            data = torch.cat((
                data,
                transform(Image.fromarray(images[index])).unsqueeze(0)
            ), dim=0)
        return data.to(next(self.parameters()).device)

    @property
    def default_transform(self) -> nn.Module:
        """Default transform for images"""
        return nn.Sequential(
            nn.ToTensor(),
            nn.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
        ) 