import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Dict, Optional
from copy import deepcopy

class iCaRLPlusFL:
    """
    Implementation of iCaRL with Federated Learning support
    Combines iCaRL's class incremental learning with federated averaging
    """
    
    def __init__(self, 
                 device: torch.device,
                 num_classes: int,
                 feature_size: int = 2048,
                 memory_size: int = 2000,
                 num_clients: int = 5) -> None:
        """
        Initialize iCaRL with Federated Learning
        
        Args:
            device: Device to run computations on
            num_classes: Total number of classes
            feature_size: Size of feature vectors
            memory_size: Maximum number of exemplars to store
            num_clients: Number of federated clients
        """
        self.device = device
        self.num_classes = num_classes
        self.feature_size = feature_size
        self.memory_size = memory_size
        self.num_clients = num_clients
        
        # Initialize client-specific storage
        self.client_exemplar_sets: Dict[int, List[torch.Tensor]] = {i: [] for i in range(num_clients)}
        self.client_exemplar_means: Dict[int, List[torch.Tensor]] = {i: [] for i in range(num_clients)}
        self.client_class_means: Dict[int, Optional[torch.Tensor]] = {i: None for i in range(num_clients)}
        
        # Global model state
        self.global_model: Optional[nn.Module] = None
        
    def _compute_exemplar_mean(self, 
                              features: torch.Tensor, 
                              indexes: torch.Tensor,
                              client_id: int) -> torch.Tensor:
        """
        Compute mean of exemplar set for a specific client
        
        Args:
            features: Feature vectors of exemplars
            indexes: Indexes of exemplars to use
            client_id: ID of the client
            
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
                             m: int,
                             client_id: int) -> None:
        """
        Construct exemplar set for a class for a specific client
        
        Args:
            images: Images to select exemplars from
            labels: Labels for the images
            class_idx: Index of the class
            m: Number of exemplars to store
            client_id: ID of the client
        """
        class_mask = labels == class_idx
        class_images = images[class_mask]
        
        with torch.no_grad():
            features = self.global_model.extract_features(class_images)
            features = F.normalize(features, p=2, dim=1)
        
        class_mean = torch.mean(features, dim=0)
        class_mean = F.normalize(class_mean, p=2, dim=0)
        
        exemplar_features = torch.zeros((m, self.feature_size)).to(self.device)
        exemplar_indexes = []
        
        for k in range(m):
            if k == 0:
                distances = torch.norm(features - class_mean.unsqueeze(0), dim=1)
                exemplar_idx = torch.argmin(distances).item()
            else:
                current_mean = torch.mean(exemplar_features[:k], dim=0)
                distances = torch.norm(
                    current_mean.unsqueeze(0) + features - class_mean.unsqueeze(0),
                    dim=1
                )
                exemplar_idx = torch.argmin(distances).item()
            
            exemplar_features[k] = features[exemplar_idx]
            exemplar_indexes.append(exemplar_idx)
            
        self.client_exemplar_sets[client_id].append(class_images[exemplar_indexes])
        
    def reduce_exemplar_sets(self, m: int, client_id: int) -> None:
        """
        Reduce size of exemplar sets for a specific client
        
        Args:
            m: Target number of exemplars per class
            client_id: ID of the client
        """
        for i in range(len(self.client_exemplar_sets[client_id])):
            self.client_exemplar_sets[client_id][i] = self.client_exemplar_sets[client_id][i][:m]
            
    def combine_dataset_with_exemplars(self, 
                                     images: torch.Tensor,
                                     labels: torch.Tensor,
                                     client_id: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Combine training dataset with stored exemplars for a specific client
        
        Args:
            images: Current training images
            labels: Current training labels
            client_id: ID of the client
            
        Returns:
            Combined images and labels including exemplars
        """
        if len(self.client_exemplar_sets[client_id]) == 0:
            return images, labels
            
        exemplar_images = torch.cat(self.client_exemplar_sets[client_id])
        exemplar_labels = torch.tensor([i for i, exemplars in enumerate(self.client_exemplar_sets[client_id]) 
                                      for _ in range(len(exemplars))]).to(self.device)
        
        all_images = torch.cat([images, exemplar_images])
        all_labels = torch.cat([labels, exemplar_labels])
        
        return all_images, all_labels
        
    def classify(self, 
                features: torch.Tensor,
                client_id: int,
                compute_means: bool = True) -> torch.Tensor:
        """
        Classify images using nearest-mean-of-exemplars rule for a specific client
        
        Args:
            features: Features to classify
            client_id: ID of the client
            compute_means: Whether to recompute class means
            
        Returns:
            Predicted class labels
        """
        if compute_means or self.client_class_means[client_id] is None:
            means = []
            for exemplars in self.client_exemplar_sets[client_id]:
                with torch.no_grad():
                    exemplar_features = self.global_model.extract_features(exemplars)
                    exemplar_features = F.normalize(exemplar_features, p=2, dim=1)
                    mean = torch.mean(exemplar_features, dim=0)
                    mean = F.normalize(mean, p=2, dim=0)
                    means.append(mean)
            self.client_class_means[client_id] = torch.stack(means)
        
        features = F.normalize(features, p=2, dim=1)
        distances = torch.cdist(features, self.client_class_means[client_id])
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
        clf_loss = F.cross_entropy(outputs, targets)
        
        old_classes = old_outputs.size(1)
        if old_classes > 0:
            dist_target = F.softmax(old_outputs[:, :old_classes] / T, dim=1)
            dist_output = F.log_softmax(outputs[:, :old_classes] / T, dim=1)
            dist_loss = -torch.mean(torch.sum(dist_target * dist_output, dim=1)) * (T ** 2)
            return clf_loss + dist_loss
            
        return clf_loss

    def federated_averaging(self, client_models: Dict[int, nn.Module]) -> None:
        """
        Perform federated averaging of client models
        
        Args:
            client_models: Dictionary of client models
        """
        global_state = self.global_model.state_dict()
        
        # Initialize with zeros
        for key in global_state.keys():
            global_state[key] = torch.zeros_like(global_state[key])
        
        # Average client models
        for client_id, model in client_models.items():
            client_state = model.state_dict()
            for key in global_state.keys():
                global_state[key] += client_state[key] / self.num_clients
        
        self.global_model.load_state_dict(global_state)

    def distribute_model(self, client_models: Dict[int, nn.Module]) -> None:
        """
        Distribute global model to all clients
        
        Args:
            client_models: Dictionary of client models
        """
        global_state = self.global_model.state_dict()
        for client_id in client_models.keys():
            client_models[client_id].load_state_dict(global_state)