#!/bin/bash

# Mistral-ORPO series are trained on 4 * A100s

accelerate launch --config_file ./src/accelerate/fsdp.yaml main.py \
    --lr 5e-6 \
    --lr_scheduler_type inverse_sqrt \
    --alpha 0.1 \
    --torch_compile False \
    --warmup_steps 200 \
    --model_name mistralai/Mistral-7B-v0.1 \
    --data_name argilla/ultrafeedback-binarized-preferences-cleaned \
    --num_train_epochs 5 \
    --prompt_max_length 1792 \
    --response_max_length 2048 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --gradient_accumulation_steps 1 \
    --num_proc 8 \
    --flash_attention_2 