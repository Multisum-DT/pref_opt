CUDA_VISIBLE_DEVICES=1 python main.py \
    --model=mistral \
    --train \
    --seed=42 \
    --level=sentence \
    --epochs=1 \
    --max_len=4096 \
    --lora_r=32 \
    --lora_alpha=64 \
    --dropout=0.1 \
    --batch_size=16 \
    --warmup_ratio=0.1 \
    --learning_rate=1e-5 \
    --weight_decay=0.01 \
    --num_save_per_epoch=100 \
    --gradient_accumulation_steps=1 \
    --eval_accumulation_steps=10 \
    --gradient_checkpointing \
    # --ckpt_dir="/data2/brian/personal/translation/checkpoints/mistral_04092203/checkpoint-50505"