
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Dict, Optional
from PIL import Image
import torchvision.transforms as transforms

class iCaRL:
    """
    Implementation of iCaRL (Incremental Classifier and Representation Learning)
    for class incremental learning with knowledge distillation and prototype rehearsal
    """
    
    def __init__(self, 
                 device: torch.device,
                 num_classes: int,
                 feature_size: int = 2048,
                 memory_size: int = 2000) -> None:
        """
        Initialize iCaRL
        
        Args:
            device: Device to run computations on
            num_classes: Total number of classes
            feature_size: Size of feature vectors
            memory_size: Maximum number of exemplars to store
        """
        self.device = device
        self.num_classes = num_classes
        self.feature_size = feature_size
        self.memory_size = memory_size
        
        # Initialize storage for exemplars
        self.exemplar_sets: List[torch.Tensor] = []
        self.exemplar_means: List[torch.Tensor] = []
        self.class_means: Optional[torch.Tensor] = None
        
    def _compute_exemplar_mean(self, 
                              features: torch.Tensor, 
                              indexes: torch.Tensor) -> torch.Tensor:
        """
        Compute mean of exemplar set
        
        Args:
            features: Feature vectors of exemplars
            indexes: Indexes of exemplars to use
            
        Returns:
            Mean feature vector
        """
        mean = torch.zeros((self.feature_size,)).to(self.device)
        for idx in indexes:
            mean += features[idx]
        mean /= len(indexes)
        mean = F.normalize(mean, p=2, dim=0)
        return mean

    def construct_exemplar_set(self,
                             images: torch.Tensor,
                             labels: torch.Tensor,
                             class_idx: int,
                             m: int) -> None:
        """
        Construct exemplar set for a class using herding selection
        
        Args:
            images: Images to select exemplars from
            labels: Labels for the images
            class_idx: Index of the class
            m: Number of exemplars to store
        """
        # Get features for all images of this class
        class_mask = labels == class_idx
        class_images = images[class_mask]
        
        with torch.no_grad():
            features = self.model.extract_features(class_images)
            features = F.normalize(features, p=2, dim=1)
        
        class_mean = torch.mean(features, dim=0)
        class_mean = F.normalize(class_mean, p=2, dim=0)
        
        # Select exemplars using herding
        exemplar_features = torch.zeros((m, self.feature_size)).to(self.device)
        exemplar_indexes = []
        
        for k in range(m):
            if k == 0:
                # Initialize with image closest to mean
                distances = torch.norm(features - class_mean.unsqueeze(0), dim=1)
                exemplar_idx = torch.argmin(distances).item()
            else:
                # Find image that makes new mean closest to class mean
                current_mean = torch.mean(exemplar_features[:k], dim=0)
                distances = torch.norm(
                    current_mean.unsqueeze(0) + features - class_mean.unsqueeze(0),
                    dim=1
                )
                exemplar_idx = torch.argmin(distances).item()
            
            exemplar_features[k] = features[exemplar_idx]
            exemplar_indexes.append(exemplar_idx)
            
        self.exemplar_sets.append(class_images[exemplar_indexes])
        
    def reduce_exemplar_sets(self, m: int) -> None:
        """
        Reduce size of exemplar sets
        
        Args:
            m: Target number of exemplars per class
        """
        for i in range(len(self.exemplar_sets)):
            self.exemplar_sets[i] = self.exemplar_sets[i][:m]
            
    def combine_dataset_with_exemplars(self, 
                                     images: torch.Tensor,
                                     labels: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Combine training dataset with stored exemplars
        
        Args:
            images: Current training images
            labels: Current training labels
            
        Returns:
            Combined images and labels including exemplars
        """
        if len(self.exemplar_sets) == 0:
            return images, labels
            
        exemplar_images = torch.cat(self.exemplar_sets)
        exemplar_labels = torch.tensor([i for i, exemplars in enumerate(self.exemplar_sets) 
                                      for _ in range(len(exemplars))])
        
        all_images = torch.cat([images, exemplar_images])
        all_labels = torch.cat([labels, exemplar_labels])
        
        return all_images, all_labels
        
    def classify(self, 
                features: torch.Tensor,
                compute_means: bool = True) -> torch.Tensor:
        """
        Classify images using nearest-mean-of-exemplars rule
        
        Args:
            features: Features to classify
            compute_means: Whether to recompute class means
            
        Returns:
            Predicted class labels
        """
        if compute_means or self.class_means is None:
            means = []
            for exemplars in self.exemplar_sets:
                with torch.no_grad():
                    exemplar_features = self.model.extract_features(exemplars)
                    exemplar_features = F.normalize(exemplar_features, p=2, dim=1)
                    mean = torch.mean(exemplar_features, dim=0)
                    mean = F.normalize(mean, p=2, dim=0)
                    means.append(mean)
            self.class_means = torch.stack(means)
        
        # Compute distances to class means
        features = F.normalize(features, p=2, dim=1)
        distances = torch.cdist(features, self.class_means)
        predictions = torch.argmin(distances, dim=1)
        
        return predictions

    def knowledge_distillation_loss(self,
                                  outputs: torch.Tensor,
                                  targets: torch.Tensor,
                                  old_outputs: torch.Tensor,
                                  T: float = 2.0) -> torch.Tensor:
        """
        Compute knowledge distillation loss
        
        Args:
            outputs: New model outputs
            targets: Ground truth targets
            old_outputs: Old model outputs
            T: Temperature for softening probability distributions
            
        Returns:
            Combined classification and distillation loss
        """
        # Classification loss for new classes
        clf_loss = F.cross_entropy(outputs, targets)
        
        # Distillation loss for old classes
        old_classes = old_outputs.size(1)
        if old_classes > 0:
            dist_target = F.softmax(old_outputs[:, :old_classes] / T, dim=1)
            dist_output = F.log_softmax(outputs[:, :old_classes] / T, dim=1)
            dist_loss = -torch.mean(torch.sum(dist_target * dist_output, dim=1)) * (T ** 2)
            return clf_loss + dist_loss
            
        return clf_loss
