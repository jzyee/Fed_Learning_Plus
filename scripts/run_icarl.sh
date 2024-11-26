#!/bin/bash

python scripts/train_icarl.py \
    --dataset_name "icifar10" \
    --img_size 32 \
    --batch_size 128 \
    --output_dir "./output/icarl" \
    --num_clients 10 \
    --local_clients 5 \
    --num_classes 2 \
    --task_size 10 \
    --tasks_global 10 \
    --memory_size 1000 \
    --epochs_local 5 \
    --epochs_global 5 \
    --learning_rate 0.1 \
    --iid_level 2 \
    --seed 42