"""
Fairness-Aware GLFC Trainer - Allocates exemplars proportionally to class sizes
"""

from typing import List, Optional, Tuple, Union
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim

from training.glfc.glfc_trainer import GLFCTrainer
from model.classification_model import ClassificationModel
# from utils.metric import accuracy
from utils.losses import SupConLoss
import torch.nn.functional as F
import random
class FAGLFCTrainer(GLFCTrainer):
    """
    Fairness-Aware GLFC Trainer class - extends GLFC to support proportional exemplar allocation
    """
    def __init__(
        self,
        model: Union[nn.Module, ClassificationModel],
        batch_size: int,
        task_size: int,
        memory_size: Optional[int],
        epochs: int,
        learning_rate: float,
        train_set: Dataset,
        device: torch.device,
        encode_model: nn.Module,
        iid_level: int,
        feature_dim: Optional[int] = None,
        client_id: int = -1
    ):
        """Initialize FA-GLFC Trainer
        
        Args:
            model: Neural network model
            batch_size: Batch size for training/testing
            task_size: Number of classes per task
            memory_size: Maximum number of exemplars to store
            epochs: Number of epochs for training
            learning_rate: Learning rate for optimization
            train_set: Training dataset
            device: Device to run computations on
            encode_model: Encoder model for prototype generation
            iid_level: Level of IID distribution
            feature_dim: Feature dimension for the model
            client_id: ID of the client (-1 for server)
        """
        super().__init__(
            model=model,
            batch_size=batch_size,
            task_size=task_size,
            memory_size=memory_size,
            epochs=epochs,
            learning_rate=learning_rate,
            train_set=train_set,
            device=device,
            encode_model=encode_model,
            iid_level=iid_level,
            feature_dim=feature_dim
        )
        
        # Initialize loss functions
        self.ce_loss = nn.CrossEntropyLoss()
        self.sup_con_loss = SupConLoss()  # You'll need to import this from utils.losses
        
        self.client_id = client_id
        self.seen_class = []  # Initialize empty list to track seen classes
        self.num_examples_per_class = {}
        self.total_examples = 0
        self._compute_class_statistics()

    def _compute_class_statistics(self) -> None:
        """Compute number of examples per class in the dataset"""
        unique_classes, class_counts = np.unique(
            self.train_dataset.targets, 
            return_counts=True
        )
        self.total_examples = len(self.train_dataset.targets)
        
        for cls, count in zip(unique_classes, class_counts):
            self.num_examples_per_class[int(cls)] = int(count)
            
        print("\nClass Distribution Statistics:")
        for cls in sorted(self.num_examples_per_class.keys()):
            count = self.num_examples_per_class[cls]
            percentage = (count / self.total_examples) * 100
            print(f"Class {cls}: {count} examples ({percentage:.2f}%)")

    def _get_train_and_test_dataloader(self, classes: List[int], is_test: bool) -> DataLoader:
        """Get DataLoader for training
        
        Args:
            classes: List of class IDs
            is_test: Ignored parameter (kept for compatibility)
        
        Returns:
            DataLoader configured for training
        """
        self.train_dataset.getTrainData(classes, [], [])
        return DataLoader(self.train_dataset, batch_size=self.batchsize, shuffle=True)

    def _get_class_data(self, class_id: int) -> np.ndarray:
        return self.train_dataset.get_image_class(class_id)

    def _construct_exemplar_set(self, class_id: int, data: np.ndarray) -> None:
        """Construct exemplar set for a class using the encode_model"""
        self.model.eval()
        features = []
        with torch.no_grad():
            for img in data:
                img_tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
                # Use encode_model for feature extraction if available
                if self.encode_model is not None:
                    feature = self.encode_model(img_tensor.unsqueeze(0).to(self.device))
                else:
                    feature = self.model.get_features(img_tensor.unsqueeze(0).to(self.device))
                feature = feature.squeeze()
                feature = feature / feature.norm()
                features.append(feature)
        
        features = torch.stack(features, dim=0)
        class_mean = features.mean(0)
        class_mean = class_mean / class_mean.norm()

        exemplar = data[((features - class_mean).norm(dim=1)).argmin()]
        self.exemplar_set[class_id] = exemplar
        self.exemplar_label_set[class_id] = class_id

    def _calculate_fair_allocation(self) -> dict:
        """Calculate fair memory allocation for each seen class"""
        if not self.memory_size:
            return {k: len(self.exemplar_set.get(k, [])) for k in self.seen_class}
            
        M = self.memory_size
        m_k = {}
        
        # Calculate proportional allocation
        for k in self.seen_class:
            class_examples = self.num_examples_per_class.get(k, 0)
            m_k[k] = round((M * class_examples) / self.total_examples)
            
        # Ensure we don't exceed memory budget due to rounding
        total_allocated = sum(m_k.values())
        if total_allocated > M:
            # Adjust allocations proportionally
            scale_factor = M / total_allocated
            for k in m_k:
                m_k[k] = round(m_k[k] * scale_factor)
        
        print("\nFair Memory Allocation:")
        for k in sorted(m_k.keys()):
            percentage = (m_k[k] / M) * 100
            print(f"Class {k}: {m_k[k]} exemplars ({percentage:.2f}% of budget)")
            
        return m_k

    def _reduce_exemplar_sets(self) -> None:
        """Reduce exemplar sets using fair allocation strategy"""
        m_k = self._calculate_fair_allocation()
        
        for class_id in self.seen_class:
            if class_id in self.exemplar_set:
                target_size = m_k.get(class_id, 0)
                self.exemplar_set[class_id] = self.exemplar_set[class_id][:target_size]
                self.exemplar_label_set[class_id] = self.exemplar_label_set[class_id][:target_size]

    def beforeTrain(self, task_id: int, group: int = 1) -> None:
        """Update exemplar sets before training using fair allocation"""
        if task_id != self.task_id_old:
            self.task_id_old = task_id
            # self.model.train()
            # self.model.to(self.device)
            
            # Update number of classes based on task_size
            self.numclass = self.task_size * (task_id + 1)
            
            # Update model architecture for new classes - following GLFC approach
            in_features = self.model.fc.in_features
            out_features = self.task_size * (task_id + 1)  # Total classes seen so far
            self.model.fc = nn.Linear(in_features, out_features).to(self.device)
            
            # Update old models if they exist
            if hasattr(self, 'old_model') and self.old_model:
                for old_m in self.old_model:
                    old_m.fc = nn.Linear(in_features, out_features).to(self.device)
            
            if group != 0:
                if self.current_class is not None:
                    self.last_class = self.current_class
                self.current_class = random.sample(
                    [x for x in range(self.numclass - self.task_size, self.numclass)],
                    self.iid_level
                )
            else:
                self.last_class = None
        
        # # Randomly sample classes for this task
        # total_classes = len(self.num_examples_per_class)
        # remaining_classes = [c for c in range(total_classes) if c not in self.seen_class]
        
        # if len(remaining_classes) < self.task_size:
        #     raise ValueError(
        #         f"Client {self.client_id}: Not enough remaining classes "
        #         f"({len(remaining_classes)}) for task size {self.task_size}"
        #     )
        
        # # Randomly sample task_size classes from remaining classes
        # self.current_class = np.random.choice(
        #     remaining_classes, 
        #     size=self.task_size, 
        #     replace=False
        # ).tolist()
        
        # print(f"\nClient {self.client_id} - Selected classes for task {task_id}: {self.current_class}")
        # print(f"Model output size: {self.numclass}")
        
        # Get training data for current classes
        self.train_loader = self._get_train_and_test_dataloader(self.current_class, False)

    def train(self, local_epoch: int, model_old: Optional[nn.Module] = None) -> None:
        """Train the model for a given number of local epochs with knowledge distillation"""
        self.model.train()
        optimizer = optim.SGD(self.model.parameters(), lr=self.learning_rate, weight_decay=0.00001)
        
        print(f"\nStarting training for task with classes: {self.current_class}")
        print(f"Model output size: {self.numclass}")
        
        for epoch in range(local_epoch):
            # Learning rate adjustment logic
            if (epoch + local_epoch * 20) % 200 == 100:
                if self.numclass == self.task_size:
                    optimizer = optim.SGD(self.model.parameters(), lr=self.learning_rate / 5, weight_decay=0.00001)
                else:
                    for p in optimizer.param_groups:
                        p['lr'] = self.learning_rate / 5
            elif (epoch + local_epoch * 20) % 200 == 150:
                if self.numclass > self.task_size:
                    for p in optimizer.param_groups:
                        p['lr'] = self.learning_rate / 25
                else:
                    optimizer = optim.SGD(self.model.parameters(), lr=self.learning_rate / 25, weight_decay=0.00001)
            elif (epoch + local_epoch * 20) % 200 == 180:
                if self.numclass == self.task_size:
                    optimizer = optim.SGD(self.model.parameters(), lr=self.learning_rate / 125, weight_decay=0.00001)
                else:
                    for p in optimizer.param_groups:
                        p['lr'] = self.learning_rate / 125

            for batch_idx, (indices, images, labels) in enumerate(self.train_loader):
                images = images.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)
                
                optimizer.zero_grad()
                
                self.model = self.model.to(self.device)
                outputs = self.model(images)
                features = self.model.get_features(images)
                
                try:
                    # Calculate losses
                    ce_loss = self.ce_loss(outputs, labels)
                    features = F.normalize(features, dim=1)
                    con_loss = self.sup_con_loss(features, labels)
                    
                    # Add knowledge distillation if old model exists
                    if model_old is not None:
                        model_old.eval()
                        with torch.no_grad():
                            old_outputs = model_old(images)
                            old_features = model_old.get_features(images)
                        
                        # Ensure old_outputs has same number of classes as current outputs
                        if old_outputs.size(1) != outputs.size(1):
                            old_outputs = F.pad(
                                old_outputs, 
                                (0, outputs.size(1) - old_outputs.size(1))
                            )
                        
                        # Knowledge distillation loss
                        T = 2.0  # Temperature parameter
                        distill_loss = nn.KLDivLoss(reduction='batchmean')(
                            F.log_softmax(outputs / T, dim=1),
                            F.softmax(old_outputs / T, dim=1)
                        ) * (T * T)
                        
                        # Feature distillation loss
                        feature_distill_loss = nn.MSELoss()(features, old_features)
                        
                        # Combine all losses
                        loss = ce_loss + con_loss + 0.5 * distill_loss + 0.1 * feature_distill_loss
                    else:
                        loss = ce_loss + con_loss
                    
                    loss.backward()
                    optimizer.step()
                    
                except RuntimeError as e:
                    print(f"Error in batch {batch_idx}:")
                    print(f"Output size: {outputs.size()}")
                    # if model_old is not None:
                    #     print(f"Old model output size: {old_outputs.size()}")
                    print(f"Labels: {labels}")
                    raise e

    def afterTrain(self, task_id: int) -> None:
        """Update exemplar sets after training using fair allocation"""
        self.model.eval()
        
        # Construct exemplar sets for current classes
        for class_id in self.current_class:
            data = self._get_class_data(class_id)
            self._construct_exemplar_set(class_id, data)
        
        # Update seen classes
        self.seen_class.extend(self.current_class)
        
        # Apply fair memory allocation
        if self.memory_size:
            self._reduce_exemplar_sets() 

    def _remap_labels(self, labels: torch.Tensor) -> torch.Tensor:
        """Remap labels to be within range [0, numclass-1]"""
        # Create mapping for current task's classes
        class_to_idx = {cls: idx for idx, cls in enumerate(sorted(self.current_class))}
        
        # Remap labels
        remapped = torch.tensor([class_to_idx[label.item()] for label in labels])
        return remapped.to(labels.device) 