import torch
import torch.nn as nn
import torch.nn.functional as F

class SupConLoss(nn.Module):
    """
    Supervised Contrastive Learning Loss: https://arxiv.org/abs/2004.11362
    This loss encourages samples from the same class to be close together in the embedding space
    while pushing samples from different classes apart, considering class frequencies.
    """
    def __init__(self, temperature: float = 0.07):
        """
        Args:
            temperature: Scaling factor for the similarity scores (default: 0.07)
        """
        super().__init__()
        self.temperature = temperature

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Calculate supervised contrastive loss
        
        Args:
            features: Feature vectors from the model (N, D) where N is batch size and D is feature dimension
            labels: Ground truth labels (N,)
            
        Returns:
            Scalar loss value
        """
        # Add validation checks
        if not torch.is_tensor(features) or not torch.is_tensor(labels):
            raise ValueError("Both features and labels must be torch tensors")
            
        if features.dim() != 2:
            raise ValueError(f"Features should be 2-dimensional but got shape {features.shape}")
            
        if labels.dim() != 1:
            raise ValueError(f"Labels should be 1-dimensional but got shape {labels.shape}")
            
        if features.size(0) != labels.size(0):
            raise ValueError(f"Number of features ({features.size(0)}) and labels ({labels.size(0)}) must match")
            
        # Check for invalid labels
        min_label = labels.min().item()
        max_label = labels.max().item()
        if min_label < 0:
            raise ValueError(f"Found negative label value: {min_label}")
            
        # Print debugging information
        print(f"Label range: [{min_label}, {max_label}]")
        print(f"Unique labels: {torch.unique(labels).tolist()}")
        
        # Normalize feature vectors
        features = F.normalize(features, dim=1)
        
        # Calculate similarity matrix
        similarity_matrix = torch.matmul(features, features.T) / self.temperature
        
        # Create mask for positive pairs (same class)
        labels = labels.view(-1, 1)
        mask = torch.eq(labels, labels.T).float()
        
        # Remove diagonal elements (self-similarity)
        mask = mask.fill_diagonal_(0)
        
        # For numerical stability
        logits_max, _ = torch.max(similarity_matrix, dim=1, keepdim=True)
        logits = similarity_matrix - logits_max.detach()
        
        # Calculate log probabilities
        exp_logits = torch.exp(logits)
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
        
        # Calculate mean log-likelihood for positive pairs
        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-8)
        
        # Return negative of mean log-likelihood
        return -mean_log_prob_pos.mean() 