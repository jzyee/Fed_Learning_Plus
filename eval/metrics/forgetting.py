from typing import Dict, List, Optional
import numpy as np
import torch

class ForgettingMetric:
    """
    Tracks and computes forgetting metrics for FCIL evaluation.
    """
    def __init__(self):
        self.best_accuracies: Dict[int, float] = {}
        self.current_accuracies: Dict[int, float] = {}
        self.history: Dict[int, List[float]] = {}  # Tracks accuracy history per class
        
    def update(
        self,
        task_id: int,
        per_class_accuracies: Dict[int, float]
    ) -> None:
        """
        Update accuracy tracking for forgetting computation.
        
        Args:
            task_id: Current task identifier
            per_class_accuracies: Dictionary of current accuracies per class
        """
        for cls, acc in per_class_accuracies.items():
            # Initialize tracking for new classes
            if cls not in self.history:
                self.history[cls] = []
                self.best_accuracies[cls] = acc
            
            # Update tracking
            self.history[cls].append(acc)
            self.best_accuracies[cls] = max(self.best_accuracies[cls], acc)
            self.current_accuracies[cls] = acc
    
    def compute(self, seen_classes: List[int]) -> Dict[str, float]:
        """
        Compute comprehensive forgetting metrics.
        
        Args:
            seen_classes: List of class IDs seen so far
            
        Returns:
            Dict containing:
                - avg_forgetting: Average forgetting across all seen classes
                - max_forgetting: Maximum forgetting observed
                - forgetting_per_class: Per-class forgetting measurements
                - stability_score: Measure of accuracy stability
        """
        forgetting_per_class = {}
        stability_scores = []
        
        for cls in seen_classes:
            if cls in self.best_accuracies:
                # Compute forgetting
                forgetting = self.best_accuracies[cls] - self.current_accuracies[cls]
                forgetting_per_class[cls] = forgetting
                
                # Compute stability score
                if len(self.history[cls]) > 1:
                    stability = np.std(self.history[cls])
                    stability_scores.append(stability)
        
        # Avoid division by zero
        if not forgetting_per_class:
            return {
                "avg_forgetting": 0.0,
                "max_forgetting": 0.0,
                "forgetting_per_class": {},
                "stability_score": 0.0
            }
            
        return {
            "avg_forgetting": np.mean(list(forgetting_per_class.values())),
            "max_forgetting": np.max(list(forgetting_per_class.values())),
            "forgetting_per_class": forgetting_per_class,
            "stability_score": np.mean(stability_scores) if stability_scores else 0.0
        } 