'''
This file contains the initialization of the 
FAGLFC model and trainers
'''
from typing import Dict, Any, List
from torch import nn
from torch.utils.data import Dataset
import copy
from training.faglfc.faglfc_model import FAGLFCModel
from training.faglfc.faglfc_trainer import FAGLFCTrainer


def init_faglfc_components(
    args: Dict[str, Any],
    feature_extractor: nn.Module,
    train_dataset: Dataset,
    encode_model: nn.Module
) -> tuple[FAGLFCModel, List[FAGLFCTrainer]]:
    """Initialize GLFC model and trainers"""
    # Initialize global model
    global_model = FAGLFCModel(
        base_model=feature_extractor,
        num_classes=args.num_classes,
        feature_dim=512,  # Adjust based on your feature extractor
        # is_feature_extractor=True,
        memory_size=args.memory_size
    ).to(args.device)
    
    # Initialize trainers for each client
    trainers: List[FAGLFCTrainer] = []
    for _ in range(args.num_clients):
        model = FAGLFCModel(
            base_model=copy.deepcopy(feature_extractor),
            num_classes=args.task_size,
            feature_dim=512,
            # is_feature_extractor=True,
            memory_size=args.memory_size
        ).to(args.device)
        
        trainer = FAGLFCTrainer(
            model=model,
            train_set=train_dataset,
            device=args.device,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            epochs=args.epochs_local,
            task_size=args.task_size,
            encode_model=encode_model,
            memory_size=args.memory_size,
            iid_level=args.iid_level
        )
        trainers.append(trainer)
        
    return global_model, trainers