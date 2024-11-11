from typing import Dict, List, Tuple, Optional
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

def compute_accuracy(
    model: nn.Module,
    data_loader: DataLoader,
    device: str = "cuda",
    task_id: Optional[int] = None
) -> Dict[str, float]:
    """
    Compute overall and per-class accuracy for FCIL evaluation.
    
    Args:
        model: Neural network model
        data_loader: DataLoader containing test data
        device: Device to run evaluation on
        task_id: Optional task identifier for tracking
    
    Returns:
        Dict containing:
            - overall_accuracy: Average accuracy across all samples
            - per_class_accuracy: Dictionary of per-class accuracies
            - current_task_accuracy: Accuracy for current task (if task_id provided)
    """
    if not isinstance(model, nn.Module):
        raise TypeError("Model must be a PyTorch Module")

    model.eval()
    correct_per_class: Dict[int, int] = {}
    total_per_class: Dict[int, int] = {}
    
    with torch.no_grad():
        for _, images, labels in data_loader:
            try:
                images = images.to(device)
                labels = labels.to(device)
                
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                
                # Update per-class statistics
                for label, pred in zip(labels, predicted):
                    label_idx = label.item()
                    total_per_class[label_idx] = total_per_class.get(label_idx, 0) + 1
                    if label == pred:
                        correct_per_class[label_idx] = correct_per_class.get(label_idx, 0) + 1
                        
            except RuntimeError as e:
                print(f"Error processing batch: {e}")
                continue
    
    # Calculate metrics
    per_class_accuracy = {
        cls: correct_per_class.get(cls, 0) / total 
        for cls, total in total_per_class.items()
    }
    
    overall_correct = sum(correct_per_class.values())
    overall_total = sum(total_per_class.values())
    
    results = {
        "overall_accuracy": overall_correct / overall_total,
        "per_class_accuracy": per_class_accuracy,
    }
    
    # Add task-specific accuracy if task_id provided
    if task_id is not None:
        task_classes = [c for c in per_class_accuracy.keys() if c // 5 == task_id]  # Assuming 5 classes per task
        task_acc = sum(per_class_accuracy[c] for c in task_classes) / len(task_classes)
        results["current_task_accuracy"] = task_acc
        
    return results 