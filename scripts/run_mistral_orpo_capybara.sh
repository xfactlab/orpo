#!/bin/bash

# Mistral-ORPO series are trained on 4 * A100s

accelerate launch --config_file ./src/accelerate/fsdp.yaml main.py \
    --lr 5e-6 \
    --torch_compile False \
    --alpha 0.05 \
    --lr_scheduler_type inverse_sqrt \
    --cache_dir /projects/hf_cache/ \
    --warmup_steps 100 \
    --model_name mistralai/Mistral-7B-v0.1 \
    --data_name argilla/distilabel-capybara-dpo-7k-binarized \
    --num_train_epochs 3 \
    --optim adamw_bnb_8bit \
    --gradient_accumulation_steps 1 \
    --prompt_max_length 1792 \
    --response_max_length 2048 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --num_proc 8 \
    --flash_attention_2