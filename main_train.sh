#!/bin/bash

echo "### START DATE=$(date)"
echo "### HOSTNAME=$(hostname)"
echo "### CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
 
# conda 환경 활성화.
source  ~/.bashrc
conda   activate   unsloth_test2
 
# cuda 11.0 환경 구성.
ml purge
ml load cuda/11.0

CUDA_VISIBLE_DEVICES=0 python train.py \
    --model=mistral \
    --train_mode=sft \
    --seed=42 \
    --level=sentence \
    --epochs=1 \
    --max_len=4096 \
    --lora_r=64 \
    --lora_alpha=32 \
    --dropout=0.05 \
    --batch_size=16 \
    --warmup_ratio=0.1 \
    --learning_rate=2e-4 \
    --weight_decay=0.01 \
    --num_save_per_epoch=100 \
    --gradient_accumulation_steps=1 \
    --eval_accumulation_steps=10 \
    --gradient_checkpointing \

echo "###"
echo "### END DATE=$(date)"