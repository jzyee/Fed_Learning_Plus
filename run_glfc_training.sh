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
  --num_clients 30 \
  --num_classes 10 \
  --device "cuda" \
  --local_clients 10 \
  --memory_size 2000 \
  --epochs_local 20 \
  --epochs_global 100 \
  --learning_rate 2.0 \
  --method "GLFC" \
  --task_size 10 \
  --tasks_global 10 \
  --iid_level 6 \
  --output_dir "./output" \
  --seed 42 