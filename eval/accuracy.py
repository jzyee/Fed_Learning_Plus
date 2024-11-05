import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict

def evaluate(
    model: nn.Module,
    testloader: DataLoader,
    device: torch.device
) -> Dict[str, float]:
    """
    Evaluate model performance
    
    Args:
        model: CNN model to evaluate
        testloader: Test data loader
        device: Device to evaluate on
        
    Returns:
        Dictionary containing accuracy and loss metrics
    """
    model.eval()
    correct = 0
    total = 0
    test_loss = 0
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for inputs, targets in testloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    accuracy = 100.0 * correct / total
    avg_loss = test_loss / len(testloader)
    
    return {"accuracy": accuracy, "loss": avg_loss}