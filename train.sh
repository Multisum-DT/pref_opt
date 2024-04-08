#!/bin/bash

echo "### START DATE=$(date)"
echo "### HOSTNAME=$(hostname)"
echo "### CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
 
# conda 환경 활성화.
source  ~/.bashrc
conda   activate   test_39
 
# cuda 11.0 환경 구성.
ml purge
ml load cuda/11.0
 
# 활성화된 환경에서 코드 실행.
CUDA_LAUNCH_BLOCKING=1 python SFT.py --train --test --batch_size 8 --epoch_size 10 --base_model "mistralai/Mistral-7B-v0.1" --wandb_key "e70e57b685f8daeeece4ae0c4eae086fc4ce4bfa" --max_len 4096 --gradient_accumulation_steps 4 --project_desc "wmt_2024" --name "lora_sftt" 
 
echo "###"
echo "### END DATE=$(date)"