from typing import Dict, Any, List, Tuple
from torch import nn
from torch.utils.data import Dataset
import copy
from training.icarl.icarl_model import iCaRLModel
from training.icarl.icarl_client import iCaRLClient
from training.icarl.icarl_server import iCaRLServer

class iCaRL:
    """
    Main iCaRL+FL class that manages server and clients
    """
    def __init__(
        self,
        args: Dict[str, Any],
        feature_extractor: nn.Module,
        train_dataset: Dataset
    ):
        """
        Initialize iCaRL+FL components
        
        Args:
            args: Configuration arguments
            feature_extractor: Base feature extractor model
            train_dataset: Training dataset
        """
        self.args = args
        self.feature_extractor = feature_extractor
        self.train_dataset = train_dataset
        
        # Initialize server
        self.server = iCaRLServer(
            device=args.device,
            learning_rate=args.learning_rate,
            num_classes=args.num_classes,
            feature_extractor=feature_extractor,
            feature_dim=512,  # Adjust based on your feature extractor
            memory_size=args.memory_size
        )
        
        # Initialize clients
        self.clients: List[iCaRLClient] = []
        self._init_clients()
        
    def _init_clients(self) -> None:
        """Initialize client instances"""
        for _ in range(self.args.num_clients):
            # Create new model instance for each client
            model = iCaRLModel(
                base_model=copy.deepcopy(self.feature_extractor),
                num_classes=self.args.task_size,
                feature_dim=512,  # Adjust based on your feature extractor
                memory_size=self.args.memory_size
            ).to(self.args.device)
            
            # Create client with model
            client = iCaRLClient(
                model=model,
                batch_size=self.args.batch_size,
                task_size=self.args.task_size,
                memory_size=self.args.memory_size,
                epochs=self.args.epochs_local,
                learning_rate=self.args.learning_rate,
                train_set=self.train_dataset,
                device=self.args.device,
                iid_level=self.args.iid_level
            )
            self.clients.append(client)
            
    def get_server(self) -> iCaRLServer:
        """Get server instance"""
        return self.server
    
    def get_clients(self) -> List[iCaRLClient]:
        """Get list of clients"""
        return self.clients
    
    def get_components(self) -> Tuple[iCaRLServer, List[iCaRLClient]]:
        """Get both server and clients"""
        return self.server, self.clients
