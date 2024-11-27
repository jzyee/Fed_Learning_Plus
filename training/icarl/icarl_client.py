import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import random
from typing import List, Optional
from training.icarl.icarl_model import iCaRLModel

class iCaRLClient:
    """
    Client implementation for iCaRL Federated Learning
    Handles local training and exemplar management
    """
    def __init__(
        self,
        model: iCaRLModel,
        batch_size: int,
        task_size: int,
        memory_size: int,
        epochs: int,
        learning_rate: float,
        train_set: torch.utils.data.Dataset,
        device: str,
        iid_level: int
    ):
        self.device = device
        self.model = model.to(self.device)
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.memory_size = memory_size
        self.task_size = task_size
        self.iid_level = iid_level
        
        self.train_dataset = train_set
        self.current_classes: Optional[List[int]] = None
        self.old_model: Optional[iCaRLModel] = None
        
    def beforeTrain(self, task_id: int, group: int) -> None:
        """
        Prepare for training on new task
        
        Args:
            task_id: ID of new task
            group: Client group (0: old, 1: new)
        """
        num_classes = self.task_size * (task_id + 1)
        self.model.incremental_learning(num_classes)
        self.model = self.model.to(self.device)
        
        if group != 0:
            # Assign random subset of new classes to client
            class_range = range(num_classes - self.task_size, num_classes)
            self.current_classes = random.sample(list(class_range), self.iid_level)
        else:
            self.current_classes = None
            
        self.train_loader = self._get_train_loader()
        
    def train(self, old_model: Optional[iCaRLModel]) -> None:
        """
        Train the client model
        
        Args:
            old_model: Previous model for knowledge distillation
        """
        self.model.train()
        self.model = self.model.to(self.device)
        
        if old_model is not None:
            self.old_model = old_model.to(self.device)
            self.old_model.eval()
            
        optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=self.learning_rate,
            momentum=0.9,
            weight_decay=5e-4
        )
        
        for epoch in range(self.epochs):
            for _, images, labels in self.train_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(images)
                
                # Classification loss for all classes
                loss = F.cross_entropy(outputs, labels)
                
                # Knowledge distillation loss for old classes
                if self.old_model is not None:
                    with torch.no_grad():
                        old_outputs = self.old_model(images)
                    old_classes = old_outputs.size(1)
                    if old_classes > 0:
                        distill_loss = F.binary_cross_entropy_with_logits(
                            outputs[:, :old_classes],
                            torch.sigmoid(old_outputs)
                        )
                        loss = loss + distill_loss
                
                loss.backward()
                optimizer.step()
                
        # Update exemplar sets after training
        self.update_exemplar_sets()
                
    def update_exemplar_sets(self) -> None:
        """Update exemplar sets for current classes"""
        if self.current_classes is None:
            return
            
        # Calculate exemplars per class
        m = self.memory_size // len(self.current_classes)
        
        # Reduce existing exemplar sets
        self.model.reduce_exemplar_sets(m)
        
        # Construct new exemplar sets
        for class_idx in self.current_classes:
            images = self.train_dataset.get_image_class(class_idx)
            self.model.construct_exemplar_set(images, m)
            
    def _get_train_loader(self) -> DataLoader:
        """Get training data loader for current classes"""
        if self.current_classes is not None:
            # Get exemplars and their labels for current classes
            exemplar_set = []
            exemplar_label_set = []
            if hasattr(self.model, 'exemplar_sets'):
                for class_idx, exemplars in enumerate(self.model.exemplar_sets):
                    exemplar_set.extend(exemplars)
                    exemplar_label_set.extend([class_idx] * len(exemplars))

            # Pass exemplars to getTrainData
            self.train_dataset.getTrainData(
                self.current_classes,
                exemplar_set,
                exemplar_label_set
            )
            
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )
        
    def log_class_assignment(self, client_index: int) -> None:
        """Log which classes this client is assigned"""
        if self.current_classes is not None:
            classes_str = ", ".join(map(str, sorted(self.current_classes)))
            print(f"Client {client_index} assigned classes: [{classes_str}]") 
        
    def update_model(self, global_model):
        """Update client model with global model parameters."""
        self.model.load_state_dict(global_model.state_dict())
    
    def train_feature_extractor(self, server_model: torch.nn.Module) -> None:
        """
        Trains the feature extractor component of the model while freezing the classifier.
        
        Args:
            server_model (torch.nn.Module): The model received from the server to train on
        """
        # Ensure we have a valid model
        if server_model is None:
            raise ValueError("Server model cannot be None")
        
        # Set model to training mode
        self.model.train()
        
        # Copy only the feature extractor parameters, ignore classifier
        feature_extractor_dict = {
            k: v for k, v in server_model.state_dict().items() 
            if "feature_extractor" in k
        }
        self.model.load_state_dict(feature_extractor_dict, strict=False)
        
        # Freeze classifier parameters - using self.model.fc instead of classifier
        for param in self.model.fc.parameters():
            param.requires_grad = False
        
        # Create optimizer for feature extractor only
        optimizer = torch.optim.SGD(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=self.learning_rate,
            momentum=0.9
        )
        
        # Training loop - update to unpack 3 values
        for batch_idx, (_, images, labels) in enumerate(self.train_loader):
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            # Zero the gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model(images)
            
            # Calculate loss
            loss = F.cross_entropy(outputs, labels)  # Using F.cross_entropy directly
            
            # Backward pass
            loss.backward()
            
            # Update weights
            optimizer.step()
            
            # Log progress (every 100 batches)
            if batch_idx % 100 == 0:
                print(f"Training batch {batch_idx}/{len(self.train_loader)}, Loss: {loss.item():.4f}")
        
        # Unfreeze classifier parameters for next phase - using self.model.fc
        for param in self.model.fc.parameters():
            param.requires_grad = True
    
    def get_feature_extractor(self):
        """Return only the feature extractor part of the model"""
        return self.model.feature_extractor
    
    def update_feature_extractor(self, server_model):
        """Update only the feature extractor part from server model"""
        # Add null check at the start of the method
        if server_model is None:
            raise ValueError("Server model cannot be None when updating feature extractor")
        
        # Copy only feature extractor parameters
        feature_extractor_dict = {
            k: v for k, v in server_model.state_dict().items() 
            if "feature_extractor" in k
        }
        self.model.load_state_dict(feature_extractor_dict, strict=False)