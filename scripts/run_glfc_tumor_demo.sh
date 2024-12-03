#!/bin/bash

# Set environment variables if needed
export CUDA_VISIBLE_DEVICES=2

# Run training script with arguments
# model_name:     Name of the model architecture to use, for options, see model/model_factory.py (e.g., ResNet18)
# encoder_name:   Name of the encoder architecture to use, for options, see encoder/encoder_factory.py (e.g., LeNet)
# dataset_name:   Name of the dataset to use for training, for options, see datasets/dataset_factory.py (e.g., mri_tumor_17)
# img_size:       resizing dims (width and height in pixels)
# batch_size:     Number of samples per training batch
# num_clients:    Total number of federated learning clients
# num_classes:    Total number of classification categories
# device:         Computing device for training (cuda only for now)
# local_clients:  Number of clients selected per round
# memory_size:    Size of exemplar memory buffer
# epochs_local:   Number of training epochs per local client
# epochs_global:  Number of global aggregation rounds
# learning_rate:  Initial learning rate for optimization
# method:         Federated Continual Learning method to use
# task_size:      Number of classes to learn per task
# tasks_global:   Total number of sequential tasks
# iid_level:      Degree of data distribution similarity across clients (1-num_classes)
# output_dir:     Directory for saving results and checkpoints
# seed:           Random seed for reproducibility

time python scripts/train_gflc.py \
  --model_name "ResNet18" \
  --encoder_name "LeNet" \
  --dataset_name "mri_tumor_17" \
  --img_size 32 \
  --batch_size 128 \
  --num_clients 10 \
  --num_classes 2 \
  --device "cuda" \
  --local_clients 5 \
  --memory_size 500 \
  --epochs_local 10 \
  --epochs_global 10 \
  --learning_rate 0.1 \
  --method "GLFC" \
  --task_size 2 \
  --tasks_global 5 \
  --iid_level 2 \
  --output_dir "output" \
  --seed 42