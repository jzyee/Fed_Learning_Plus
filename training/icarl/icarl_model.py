from typing import Optional, List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import numpy as np
from model.classification_model import ClassificationModel

class iCaRLModel(ClassificationModel):
    """
    iCaRL model with exemplar management as per original iCaRL paper
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
            feature_dim=feature_dim
        )
        self.memory_size = memory_size
        self.exemplar_sets: List[List[torch.Tensor]] = []  # P = {P₁, P₂, ..., Pₖ}
        self.class_means: List[torch.Tensor] = []  # Class means for NME classification
        
        # Add device initialization
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        
    def construct_exemplar_set(self, images: torch.Tensor, m: int) -> List[torch.Tensor]:
        """
        Construct exemplar set P for a class using herding selection
        Algorithm 4 in the paper
        
        Args:
            images: Images of the class
            m: Number of exemplars to store
        """
        features = self.extract_features(images)  # ϕ(x_i)
        class_mean = features.mean(0)  # μ = 1/n ∑ϕ(x_i)
        class_mean = class_mean / class_mean.norm()  # Normalize
        
        # Select exemplars
        exemplars = []
        exemplar_features = torch.zeros_like(features[0]).unsqueeze(0)
        
        for k in range(m):
            if len(exemplars) == 0:
                S = features  # ϕ(x)
            else:
                S = features + exemplar_features.sum(0, keepdim=True) / (k + 1)
            
            # Select argmin ‖μ - 1/(k+1)(S + ϕ(x))‖
            S = S / S.norm(dim=1, keepdim=True)  # Normalize
            distances = (class_mean.unsqueeze(0) - S).norm(2, 1)
            idx = distances.argmin().item()
            
            exemplars.append(images[idx])
            exemplar_features = torch.cat([exemplar_features, features[idx].unsqueeze(0)])
            
            # Remove selected feature to avoid duplicates
            mask = torch.ones(features.shape[0], dtype=torch.bool)
            mask[idx] = False
            features = features[mask]
            images = images[mask]
        
        return exemplars

    def reduce_exemplar_sets(self, m: int) -> None:
        """
        Reduce exemplar set sizes (Algorithm 4 in paper)
        
        Args:
            m: New size for each exemplar set
        """
        for i in range(len(self.exemplar_sets)):
            self.exemplar_sets[i] = self.exemplar_sets[i][:m]

    def combine_dataset_with_exemplars(self, dataset: torch.utils.data.Dataset) -> torch.utils.data.Dataset:
        """
        Combine current dataset with exemplars (Algorithm 5 in paper)
        """
        for exemplar_set in self.exemplar_sets:
            dataset.add_exemplars(exemplar_set)
        return dataset

    def update_representation(self, dataset: torch.utils.data.Dataset) -> None:
        """
        Update feature representation (Algorithm 3 in paper)
        """
        # Merge current exemplars with new data
        dataset = self.combine_dataset_with_exemplars(dataset)
        
        # Train with classification and distillation loss
        optimizer = torch.optim.SGD(self.parameters(), lr=0.1, momentum=0.9)
        
        for epoch in range(70):  # As per paper
            for images, labels in dataset:
                optimizer.zero_grad()
                outputs = self(images)
                
                # Classification loss for new classes
                classification_loss = F.cross_entropy(outputs, labels)
                
                # Distillation loss for old classes
                if self.old_model is not None:
                    with torch.no_grad():
                        old_outputs = self.old_model(images)
                    distillation_loss = F.binary_cross_entropy_with_logits(
                        outputs[:, :old_outputs.size(1)],
                        torch.sigmoid(old_outputs)
                    )
                    loss = classification_loss + distillation_loss
                else:
                    loss = classification_loss
                
                loss.backward()
                optimizer.step()

    def classify(self, images: torch.Tensor) -> torch.Tensor:
        """
        Nearest-Mean-of-Exemplars classification (Algorithm 2 in paper)
        """
        features = self.extract_features(images)
        features = F.normalize(features, dim=1)  # L2 normalization
        
        # Compute distances to class means
        means = torch.stack(self.class_means)
        distances = torch.cdist(features, means)
        
        return distances.argmin(dim=1)

    def extract_features(self, images):
        """
        Extract features from images using the feature extractor
        
        Args:
            images: Input images to extract features from
        
        Returns:
            torch.Tensor: Extracted features
        """
        # Convert numpy array to torch tensor if necessary
        if isinstance(images, np.ndarray):
            images = torch.from_numpy(images).float()
        
        # Ensure the input is on the correct device and in the correct format
        images = images.to(self.device)
        
        # Convert from NHWC to NCHW format if necessary
        if images.shape[-1] == 3:  # If channels are last
            images = images.permute(0, 3, 1, 2)
        
        # Extract features
        with torch.no_grad():
            features = self.feature_extractor(images)
        return F.normalize(features, dim=1) 

    def incremental_learning(self, num_classes: int) -> None:
        """
        Extends the model's classifier to accommodate new classes.
        
        Args:
            num_classes: Number of new classes to add to the classifier
        """
        in_features = self.fc.in_features
        out_features = self.fc.out_features
        
        # Create new classifier with extended output size
        weight = self.fc.weight.data
        bias = self.fc.bias.data
        
        self.fc = nn.Linear(in_features, num_classes)
        
        # If there were previous classes, copy their weights
        if out_features > 0:
            self.fc.weight.data[:out_features] = weight
            self.fc.bias.data[:out_features] = bias