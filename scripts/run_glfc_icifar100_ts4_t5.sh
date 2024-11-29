#!/bin/bash

# Set environment variables if needed
export CUDA_VISIBLE_DEVICES=3

# Run training script with arguments
time python scripts/train_gflc.py \
  --model_name "ResNet18" \
  --encoder_name "LeNet" \
  --dataset_name "icifar100" \
  --img_size 32 \
  --batch_size 128 \
  --num_clients 10 \
  --num_classes 4 \
  --device "cuda" \
  --local_clients 2 \
  --memory_size 2000 \
  --epochs_local 10 \
  --epochs_global 10 \
  --learning_rate 0.1 \
  --method "GLFC" \
  --task_size 4 \
  --tasks_global 5 \
  --iid_level 3 \
  --output_dir "output" \
  --seed 42