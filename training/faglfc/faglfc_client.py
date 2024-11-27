from typing import List, Optional, Tuple
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader

from training.base import BaseFedClient
from models.base import BaseLearner
from utils.losses import SupConLoss

class FAGLFCClient(BaseFedClient):
    def __init__(
        self,
        client_id: int,
        model: BaseLearner,
        args: dict,
        train_loader: DataLoader,
        test_loader: DataLoader,
        device: torch.device,
        init_local_epoch: int
    ):
        super().__init__(
            client_id, model, args, train_loader, 
            test_loader, device, init_local_epoch
        )
        self.ce_loss = nn.CrossEntropyLoss()
        self.sup_con_loss = SupConLoss(temperature=0.07)
        self.exemplar_set = {}
        self.exemplar_label_set = {}
        self.seen_class = []
        self.current_class = []
        self.current_task = 0

    def train(self) -> Tuple[float, float]:
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for _, (indices, images, labels) in enumerate(self.train_loader):
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(images)
            features = self.model.extract_vector(images)
            
            ce_loss = self.ce_loss(outputs, labels)
            con_loss = self.sup_con_loss(features, labels)
            loss = ce_loss + self.args.get("lambda_con", 1.0) * con_loss
            
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        accuracy = 100. * correct / total
        avg_loss = total_loss / len(self.train_loader)
        
        return avg_loss, accuracy 