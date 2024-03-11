#!/bin/bash

#SBATCH --time=24:00:00
#SBATCH --cpus-per-gpu=8


accelerate launch --config_file ./src/accelerate/ds2.yaml main.py \
    --lr 5e-6 \
    --warmup_steps 100 \
    --model_name facebook/opt-1.3b \
    --data_name HuggingFaceH4/ultrafeedback_binarized \
    --num_train_epochs 5 \
    --prompt_max_length 128 \
    --response_max_length 2048 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --num_proc 8