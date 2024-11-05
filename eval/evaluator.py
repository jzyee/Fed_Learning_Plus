from typing import Dict, List, Optional
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import logging

from .metrics.accuracy import compute_accuracy
from .metrics.entropy import EntropyMetric
from .metrics.forgetting import ForgettingMetric

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FCILEvaluator:
    """
    Comprehensive evaluator for Few-Shot Class Incremental Learning.
    """
    def __init__(
        self,
        model: nn.Module,
        device: str = "cuda",
        entropy_threshold: float = 1.2
    ):
        """
        Args:
            model: Neural network model to evaluate
            device: Device to run evaluation on
            entropy_threshold: Threshold for entropy-based compensation
        """
        self.model = model
        self.device = device
        self.forgetting_metric = ForgettingMetric()
        self.entropy_metric = EntropyMetric(threshold=entropy_threshold)
        self.results_history: List[Dict] = []
        
    def evaluate_task(
        self,
        task_id: int,
        test_loader: DataLoader,
        seen_classes: List[int],
        current_task_classes: Optional[List[int]] = None
    ) -> Dict:
        """
        Evaluate model performance on current task.
        
        Args:
            task_id: Current task identifier
            test_loader: DataLoader for testing
            seen_classes: List of class IDs seen so far
            current_task_classes: Optional list of classes in current task
            
        Returns:
            Dict containing comprehensive evaluation metrics
        """
        try:
            # Compute accuracy metrics
            accuracy_metrics = compute_accuracy(
                self.model,
                test_loader,
                self.device,
                task_id
            )
            
            # Update forgetting tracking
            self.forgetting_metric.update(
                task_id,
                accuracy_metrics["per_class_accuracy"]
            )
            forgetting_metrics = self.forgetting_metric.compute(seen_classes)
            
            # Compute entropy metrics
            entropy_metrics = self.entropy_metric.compute(
                self.model,
                test_loader,
                self.device
            )
            
            # Compile results
            results = {
                "task_id": task_id,
                **accuracy_metrics,
                **forgetting_metrics,
                **entropy_metrics,
                "seen_classes": len(seen_classes),
                "timestamp": torch.cuda.Event(enable_timing=True)
            }
            
            # Add to history
            self.results_history.append(results)
            
            return results
            
        except Exception as e:
            logger.error(f"Error during evaluation: {e}")
            raise
            
    def get_summary(self) -> Dict:
        """
        Get summary statistics across all evaluated tasks.
        """
        if not self.results_history:
            return {}
            
        return {
            "final_accuracy": self.results_history[-1]["overall_accuracy"],
            "peak_accuracy": max(r["overall_accuracy"] for r in self.results_history),
            "final_forgetting": self.results_history[-1]["avg_forgetting"],
            "total_tasks_evaluated": len(self.results_history),
            "compensation_triggers": sum(
                1 for r in self.results_history if r["compensation_needed"]
            )
        }