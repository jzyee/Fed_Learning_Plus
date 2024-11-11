from dataclasses import dataclass, field
import argparse
from model.model_factory import load_model
from dataset.dataset_factory import load_dataset
from encoder.encoder_factory import load_encoder
from dataclasses import fields
from transformers import HfArgumentParser
from utils.utils import setup_seed, weights_init
from utils.img_transforms import train_transforms, test_transforms
from model.classification_model import ClassificationModel
from training.training_factory import load_training_method
from training.glfc.glfc_proxy_server import ProxyServer
import torch
import os
import os.path as osp
import random
from utils.fed_utils import local_train, participant_exemplar_storing, model_global_eval
import copy
from weight_agg.fed_avg import fed_avg

# set data class for args
@dataclass
class ModelArgs:
    model_name: str = field(
        default="ResNet18",
        metadata={"help": "model to use for training"})

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
    

@dataclass
class TrainingArgs:
    batch_size: int = field(
        default=128,
        metadata={"help": "batch size"})
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
    output_dir: str = field(
        default="./output",
        metadata={"help": "output directory"})




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



# set the output directory
output_dir = osp.join(training_args.output_dir, f"{model_args.model_name}_{encoder_args.encoder_name}_{dataset_args.dataset_name}_seed{training_args.seed}")
if not osp.exists(output_dir):
    os.makedirs(output_dir, exist_ok=True)


out_file = open(osp.join(output_dir, 'log_tar_' + str(training_args.task_size) + '.txt'), 'w')
log_str = 'method_{}, task_size_{}, learning_rate_{}'.format(training_args.method, training_args.task_size, training_args.learning_rate)
out_file.write(log_str + '\n')
out_file.flush()

# init model
model = load_model(model_args.model_name)

# set the parameters for learning

# old clients for the old task
old_clients_0 = []
# old clients for the new task
old_clients_1 = [i for i in range(training_args.num_classes)]
# new clients for the new task
new_clients = []
# list of old models
model_list = []


# seed training with the number, seed, in training args
setup_seed(training_args.seed)

# # model settings
# model_global = ClassificationModel(model, training_args.num_classes)
# model_global.to(training_args.device)

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

# initialize the global model and trainers(which include local models for each client)
# global model is a GLFCModel
global_model, trainers = load_training_method(training_args, 
                                              model, 
                                              train_dataset, 
                                              encode_model)

print(f"no. of trainers: {len(trainers)}")

# initialize the proxy server
proxy_server = ProxyServer(
    training_args.device,
    training_args.learning_rate,
    training_args.num_classes,
    global_model,
    encode_model,
    test_transform,
    512 # feature dimension for resnet18 is 512
    )

classes_learned = training_args.task_size
old_task_id = -1

# Calculate epochs per task based on total epochs and number of tasks
epochs_per_task = training_args.epochs_global // training_args.tasks_global
print(f"\nTraining Configuration:")
print(f"Total epochs: {training_args.epochs_global}")
print(f"Total tasks: {training_args.tasks_global}")
print(f"Epochs per task: {epochs_per_task}")
print(f"Classes per task: {training_args.task_size}\n")

for epoch_g in range(training_args.epochs_global):
    print(f"\nGlobal epoch {epoch_g + 1}/{training_args.epochs_global}")
    pool_grad = []
    model_old = proxy_server.model_back()
    
    # Calculate current task based on epoch
    task_id = epoch_g // epochs_per_task
    
    # Print progress information
    print(f"Current task: {task_id + 1}/{training_args.tasks_global}")
    print(f"Epochs remaining in current task: {epochs_per_task - (epoch_g % epochs_per_task)}")

    # if the task is new, update the old clients for the new task
    if task_id != old_task_id and old_task_id != -1:
        # update the old clients for the new task
        overall_client = len(old_clients_0) + len(old_clients_1) + len(new_clients)
        # get the new client classes for the new task
        new_clients = [i for i in range(overall_client, overall_client + training_args.task_size)]
        # get the old client classes for the new task
        # 90% of the old clients are used for the new task
        old_clients_1 = random.sample([i for i in range(overall_client)], int(overall_client * 0.9))
        # the remaining 10% of the old clients are used for the old task
        old_clients_0 = [i for i in range(overall_client) if i not in old_clients_1]
        # update the number of clients
        num_clients = len(new_clients) + len(old_clients_1) + len(old_clients_0)
        print(f"Task transition: Reassigning clients")
        print(f"New clients: {len(new_clients)}, Old clients (new task): {len(old_clients_1)}, Old clients (old task): {len(old_clients_0)}")

        classes_learned += training_args.task_size
        global_model.incremental_learning(task_id)
    
    print(f"federated global round {epoch_g + 1}/{training_args.epochs_global}, task {task_id + 1}/{training_args.tasks_global}, classes seen: {classes_learned}")

    # list of local models
    w_local = []
    clients_index = random.sample(range(training_args.num_clients), training_args.local_clients)
    print(f"no. of local clients: {training_args.local_clients}")
    print(f"selected clients: {clients_index}")

    for c in clients_index:
        local_model, proto_grad = local_train(trainers, c, global_model, task_id, model_old, epoch_g, old_clients_0)
        w_local.append(local_model)
        if proto_grad is not None:
            for grad_i in proto_grad:
                pool_grad.append(grad_i)

    print('update every participant/client examplar set and old model')
    participant_exemplar_storing(trainers, training_args.local_clients, global_model, old_clients_0, task_id, clients_index )
    print('update completed')

    

    print('federated aggregation start')
    w_g_new = fed_avg(w_local)
    w_g_last = copy.deepcopy(global_model.state_dict())
    print('federated aggregation completed')

    global_model.load_state_dict(w_g_new)

    print('set up an exemplar set and old model')
    proxy_server.model = copy.deepcopy(global_model)
    proxy_server.dataloader(pool_grad)
    print('set up completed')
    
    acc_global = model_global_eval(global_model, test_dataset, task_id, training_args.task_size, training_args.device)
    log_str = 'Task: {}, Round: {} Accuracy = {:.2f}%'.format(task_id, epoch_g, acc_global)
    out_file.write(log_str + '\n')
    out_file.flush()
    print('classification accuracy of global model at round %d: %.3f \n' % (epoch_g, acc_global))

    old_task_id = task_id

# # Run federated learning process across multiple tasks
# for task_id in range(training_args.tasks_global):
#     print(f"\nStarting task {task_id + 1}/{training_args.tasks_global}")
    
#     # Get classes for current task
#     ## get the start and end classes for the current task
#     start_class = task_id * training_args.task_size
#     end_class = (task_id + 1) * training_args.task_size
#     ## get the classes for the current task
#     task_classes = list(range(start_class, end_class))
    
#     # Update dataset for current task classes
#     for trainer in trainers:
#         ## update the train loader for the current task
#         ## this is done by updating the dataset for the trainer
#         trainer.update_train_loader(task_classes)
    
#     # Train local models on clients
#     for epoch in range(training_args.epochs_global):
#         print(f"\nGlobal epoch {epoch + 1}/{training_args.epochs_global}")
        
#         # Local training on each client
#         for client_id, trainer in enumerate(trainers):
#             print(f"Training client {client_id + 1}/{len(trainers)}")
            
#             # Get old model states for knowledge distillation
#             old_models = proxy_server.get_old_models()
            
#             # Train client model
#             trainer.train(epoch, old_models)
            
#             # Get gradients from client training
#             client_grads = trainer.get_gradients()
            
#             # Send gradients to proxy server
#             proxy_server.receive_gradients(client_id, client_grads)
        
#         # Proxy server processes gradients and updates global model
#         proxy_server.update_global_model()
        
#         # Check convergence through proxy monitoring
#         if proxy_server.check_convergence():
#             print("Training converged early")
#             break
    
#     # Update global model for next task
#     proxy_server.prepare_next_task()

# print("\nTraining completed!")

# # Save final model
# torch.save(global_model.state_dict(), f"{training_args.output_dir}/final_model.pt")



