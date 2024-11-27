from typing import Dict, List, Optional
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import copy
from training.icarl.icarl_model import iCaRLModel

class iCaRLServer:
    """
    Server for iCaRL Federated Learning
    Manages global model and coordinates client training
    """
    def __init__(
        self,
        device: str,
        learning_rate: float,
        num_classes: int,
        feature_extractor: nn.Module,
        feature_dim: int,
        memory_size: int = 2000
    ):
        self.device = device
        self.learning_rate = learning_rate
        self.model = iCaRLModel(
            base_model=feature_extractor,
            num_classes=num_classes,
            feature_dim=feature_dim,
            memory_size=memory_size
        ).to(device)
        
        self.best_model = None
        self.best_acc = 0.0
        
    def aggregate_models(self, client_models: Dict[int, nn.Module]) -> None:
        """
        Aggregate client models using FedAvg
        
        Args:
            client_models: Dictionary of client models to aggregate, where keys are client IDs
                          and values are their respective models
        
        Raises:
            TypeError: If client_models is empty or contains non-nn.Module objects
            ValueError: If client_models dictionary is empty
        """
        # Validate input
        if not client_models:
            raise ValueError("client_models dictionary cannot be empty")
        
        if not all(isinstance(model, nn.Module) for model in client_models.values()):
            raise TypeError("All client models must be instances of nn.Module")
        
        # Convert parameters to float before averaging
        averaged_params = [
            torch.stack([
                client_model.state_dict()[key].float()
                for client_model in client_models.values()
            ]).mean(0)
            for key in self.model.state_dict().keys()
        ]
        
        # Update global model
        for param, key in zip(averaged_params, self.model.state_dict().keys()):
            self.model.state_dict()[key].data.copy_(param)
        
    def evaluate(self, test_loader: DataLoader) -> float:
        """
        Evaluate current model performance using NME classification
        
        Args:
            test_loader: Test data loader
            
        Returns:
            Classification accuracy
        
        Raises:
            ValueError: If test_loader is empty
        """
        # Validate test_loader
        if len(test_loader) == 0:
            raise ValueError("test_loader is empty - no batches available for evaluation")
        
        self.model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in test_loader:
                if images.size(0) == 0:
                    print(f"Warning: Encountered empty batch - images shape: {images.shape}")
                    continue
                
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                # Use NME classification as per iCaRL paper
                predictions = self.model.classify(images)
                correct += (predictions == labels).sum().item()
                total += labels.size(0)
        
        if total == 0:
            raise ValueError("No samples were processed during evaluation")
        
        accuracy = 100.0 * correct / total
        
        # Save best model
        if accuracy > self.best_acc:
            self.best_acc = accuracy
            self.best_model = copy.deepcopy(self.model)
            
        return accuracy
    
    def get_current_model(self) -> iCaRLModel:
        """
        Return current global model for client training
        
        Returns:
            iCaRLModel: A deep copy of the current model
        
        Raises:
            RuntimeError: If the current model is None
        """
        if self.model is None:
            raise RuntimeError("Current model is None - server may not be properly initialized")
        return copy.deepcopy(self.model)
    
    def get_best_model(self) -> Optional[iCaRLModel]:
        """Return best model for knowledge distillation"""
        return self.best_model 
    
    def aggregate_feature_extractors(self, client_feature_extractors):
        """Aggregate feature extractors from clients using FedAvg"""
        # Initialize aggregated parameters dictionary
        aggregated_state_dict = {}
        
        # Get the state dict of the first client as reference
        first_client = next(iter(client_feature_extractors.values()))
        reference_state_dict = first_client.state_dict()

        # Aggregate parameters
        for key in reference_state_dict.keys():
            # Collect same parameter from all clients
            params = [
                client.state_dict()[key].float() # Convert to float
                for client in client_feature_extractors.values()
            ]
            
            # Calculate mean of parameters
            aggregated_state_dict[key] = torch.stack(params).mean(dim=0)
            
            # Convert back to original dtype if needed
            if reference_state_dict[key].dtype != torch.float:
                aggregated_state_dict[key] = aggregated_state_dict[key].to(
                    dtype=reference_state_dict[key].dtype
                )

        # Load aggregated parameters back to server model
        self.model.feature_extractor.load_state_dict(aggregated_state_dict)
    
    def update_classifier(self):
        """Update the classifier using global data"""
        # Train classifier using global/proxy data
        self.model.train()
        # Only update classifier parameters
        for param in self.model.feature_extractor.parameters():
            param.requires_grad = False
        for param in self.model.fc.parameters():  # Assuming 'fc' is the classification layer
            param.requires_grad = True
        
        # Train classifier using global data or exemplars
        # ... training logic for classifier ...