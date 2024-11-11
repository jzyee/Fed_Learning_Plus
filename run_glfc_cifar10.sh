#!/bin/bash

# Set environment variables if needed
export CUDA_VISIBLE_DEVICES=0,1,2,3

# Run training script with arguments
python train_model.py \
  --model_name "ResNet18" \
  --encoder_name "LeNet" \
  --dataset_name "icifar10" \
  --img_size 32 \
  --batch_size 128 \
  --num_clients 10 \
  --num_classes 2 \
  --device "cuda" \
  --local_clients 5 \
  --memory_size 1000 \
  --epochs_local 5 \
  --epochs_global 5 \
  --learning_rate 0.1 \
  --method "GLFC" \
  --task_size 2 \
  --tasks_global 5 \
  --iid_level 2 \
  --output_dir "./output" \
  --seed 42