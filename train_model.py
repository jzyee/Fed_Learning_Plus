from dataclasses import dataclass, field
import argparse
from model.model_factory import load_model
from dataset.dataset_factory import load_dataset
from encoder.encoder_factory import load_encoder
from dataclasses import fields
from transformers import HfArgumentParser
from utils.utils import setup_seed, weights_init
from utils.img_transforms import train_transforms, test_transforms
from model.model_wrapper import ModelWrapper

# set data class for args
@dataclass
class ModelArgs:
    model_name: str = field(
        default="ResNet18",
        metadata={"help": "model to use for training"})
    seed: int = field(
        default=42,
        metadata={"help": "random seed for training"})

@dataclass
class EncoderArgs:
    encoder_name: str = field(
        default="LeNet",
        metadata={"help": "encoder to use for training"})

@dataclass
class DatasetArgs:
    dataset_name: str = field(
        default="icifar100",
        metadata={"help": "dataset to use for training"})
    img_size: int = field(
        default=32,
        metadata={"help": "image size"})
    batch_size: int = field(
        default=128,
        metadata={"help": "batch size"})

@dataclass
class TrainingArgs:
    seed: int = field(
        default=42,
        metadata={"help": "random seed for training"})
    num_clients: int = field(
        default=30,
        metadata={"help": "number of clients"})
    num_classes: int = field(
        default=10,
        metadata={"help": "number of data classes in the first task"})
    device: str = field(
        default="cuda",
        metadata={"help": "device to use for training"})
    local_clients: int = field(
        default=10,
        metadata={"help": "number of local clients"})
    memory_size: int = field(
        default=2000,
        metadata={"help": "size of examplar memory"})
    epochs_local: int = field(
        default=20,
        metadata={"help": "number of local epochs of each global round"})
    epochs_global: int = field(
        default=100,
        metadata={"help": "number of global epochs"} )
    learning_rate: float = field(
        default=2.0,
        metadata={"help": "learning rate"})
    method: str = field(
        default="GLFC",
        metadata={"help": "method to use for training"})
    task_size: int = field(
        default=10,
        metadata={"help": "number of data classes per task"})
    tasks_global: int = field(
        default=10,
        metadata={"help": "total number of tasks"})
    iid_level: int = field(
        default=6,
        metadata={"help": "number of data classes per local client"})




# init parser for model, dataset, and training parameters
parser = HfArgumentParser(
    (
        ModelArgs,
        EncoderArgs,
        DatasetArgs,
        TrainingArgs
    )
)

# take in arguements for model, dataset, and training parameters
# parse args from command line that fits the data class
model_args, encoder_args, dataset_args, training_args = parser.parse_args_into_dataclasses()

# init model
model = load_model(model_args.model_name)

# set the parameters for learning
old_client_0 = []
old_client_1 = [i for i in range(training_args.num_classes)]
new_client = []
model_list = []


# seed training with the number, seed, in training args
setup_seed(training_args.seed)

# model settings
model_global = ModelWrapper(model, training_args.num_classes)
model_global.to(training_args.device)

# define the transformations for the training and test data
train_transform = train_transforms(dataset_args.img_size)
test_transform = test_transforms(dataset_args.img_size)

# load the dataset with train and test transformations
train_dataset, test_dataset = load_dataset(
    dataset_args, train_transform, test_transform)

total_num_classes = training_args.tasks_global * training_args.task_size # 10 * 10

# init encoder model architecture
encode_model = load_encoder(encoder_args, training_args)
# init weights for encoder
encode_model.apply(weights_init)

for i in range(125):
    model_temp = GLFC_model(args.numclass, model, args.batch_size, args.task_size, args.memory_size,
                 args.epochs_local, args.learning_rate, train_dataset, args.device, encode_model)
    models.append(model_temp)