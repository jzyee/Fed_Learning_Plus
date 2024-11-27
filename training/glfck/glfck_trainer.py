"""
GLFCK Trainer - Modified version of GLFC that uses K exemplars instead of 1
"""

from typing import List, Optional, Tuple, Dict, Any
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np

from training.base import BaseTrainer
from models.base import BaseLearner
from utils.metric import accuracy
from utils.losses import SupConLoss

class GLFCKTrainer(BaseTrainer):
    """
    GLFCK Trainer class - extends GLFC to support K exemplars per class
    """
    def __init__(
        self,
        model: BaseLearner,
        args: Dict[str, Any],
        trainset: Any,
        testset: Any,
        device: torch.device,
        client_id: int = -1
    ) -> None:
        """
        Initialize GLFCK Trainer with K exemplars support
        
        Args:
            model: Base learner model
            args: Training arguments including K exemplars count
            trainset: Training dataset
            testset: Test dataset 
            device: PyTorch device
            client_id: Client identifier
        """
        super().__init__(model, args, trainset, testset, device)
        self.client_id = client_id
        self.current_task = 0
        self.current_class = []
        self.seen_class = []
        self.k_exemplars = args.get("k_exemplars", 5)  # Default to 5 if not specified
        
        # Initialize exemplar sets
        self.exemplar_set: Dict[int, torch.Tensor] = {}
        self.exemplar_label_set: Dict[int, torch.Tensor] = {}
        
        # Loss functions
        self.ce_loss = nn.CrossEntropyLoss()
        self.sup_con_loss = SupConLoss(temperature=args.get("temperature", 0.07))

    def _extract_features(self, samples: torch.Tensor) -> torch.Tensor:
        """Extract features from samples using the model"""
        self.model.eval()
        features = []
        with torch.no_grad():
            for sample in samples:
                feature = self.model.extract_vector(sample.unsqueeze(0).to(self.device))
                feature = feature.squeeze()
                feature = feature / feature.norm()
                features.append(feature)
        features = torch.stack(features, dim=0)
        return features

    def _construct_exemplar_set(self, class_id: int, data: torch.Tensor) -> None:
        """
        Construct exemplar set for a class using K samples
        
        Args:
            class_id: Class identifier
            data: Data tensor for the class
        """
        features = self._extract_features(data)
        class_mean = features.mean(0)
        class_mean = class_mean / class_mean.norm()

        exemplars = []
        exemplar_features = torch.zeros_like(features)
        
        for k in range(min(self.k_exemplars, len(data))):
            S = features + exemplar_features
            phi = S / (k + 1)
            phi = phi / phi.norm(dim=1).reshape(-1, 1)
            
            m = (phi - class_mean).norm(dim=1)
            i = m.argmin().item()
            
            exemplars.append(data[i])
            exemplar_features[k] = features[i]
            
            # Remove selected sample to avoid duplicates
            mask = torch.ones(len(data), dtype=torch.bool)
            mask[i] = False
            data = data[mask]
            features = features[mask]

        self.exemplar_set[class_id] = torch.stack(exemplars)
        self.exemplar_label_set[class_id] = torch.full(
            (len(exemplars),), 
            class_id, 
            dtype=torch.long
        )

    def _reduce_exemplar_sets(self, m: int) -> None:
        """Reduce exemplar sets to maintain fixed memory"""
        for key in self.exemplar_set.keys():
            self.exemplar_set[key] = self.exemplar_set[key][:m]
            self.exemplar_label_set[key] = self.exemplar_label_set[key][:m]

    def beforeTrain(self, task_id: int, local_epoch: int = 1) -> None:
        """Setup before training task"""
        self.current_task = task_id
        self.model.train()
        self.model.to(self.device)
        
        # Get train loader for current classes
        self.train_loader = self._get_train_and_test_dataloader(
            self.current_class, 
            False
        )

    def train(self, local_epoch: int) -> None:
        """Train for local epochs"""
        self.model.train()
        for _ in range(local_epoch):
            for _, (indices, images, labels) in enumerate(self.train_loader):
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                self.optimizer.zero_grad()
                
                # Get model outputs
                outputs = self.model(images)
                features = self.model.extract_vector(images)
                
                # Calculate losses
                ce_loss = self.ce_loss(outputs, labels)
                con_loss = self.sup_con_loss(features, labels)
                
                # Combined loss
                loss = ce_loss + self.args.get("lambda_con", 1.0) * con_loss
                
                loss.backward()
                self.optimizer.step()

    def afterTrain(self, task_id: int) -> None:
        """Update exemplar sets after training"""
        self.model.eval()
        for class_id in self.current_class:
            data = self._get_class_data(class_id)
            self._construct_exemplar_set(class_id, data)
            
        # Update seen classes
        self.seen_class.extend(self.current_class)
        
        # Reduce exemplar sets if needed
        if self.args.get("memory_budget"):
            m = self.args["memory_budget"] // len(self.seen_class)
            self._reduce_exemplar_sets(m)

    def _get_class_data(self, class_id: int) -> torch.Tensor:
        """Get all data for a specific class"""
        class_mask = self.trainset.targets == class_id
        return self.trainset.data[class_mask] 