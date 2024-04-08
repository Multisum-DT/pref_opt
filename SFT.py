from utils import *
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer
)

from peft import (
    LoraConfig, 
    PeftModel, 
    prepare_model_for_kbit_training, 
    get_peft_model
)

import wandb

from unsloth import FastLanguageModel
from trl import SFTTrainer,DataCollatorForCompletionOnlyLM
import argparse
import torch
import numpy as np
import random
from datetime import datetime
#from evaluate import load
from datasets import load_metric

import ipdb
import warnings
warnings.filterwarnings('ignore')

def parse_args():

    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model",type=str,required=True)
    parser.add_argument("--dataset_path",type=str, default="./wmt_utils.py")
    parser.add_argument("--wandb_key",type=str)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--lora_r", type=int, default=32)
    parser.add_argument("--lora_alpha", type=int, default=64)
    parser.add_argument("--dropout_rate", type=float, default=0.1)
    parser.add_argument("--max_len", type=int, default=2048)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--epoch_size', type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps",type=int,default=2)
    parser.add_argument("--gradient_checkpointing",type=bool,default=False,help="reduce required memory size but slower training")
    parser.add_argument("--ckpt_path",type=str,default=None)
    parser.add_argument("--train",action="store_true")
    parser.add_argument("--test",action="store_true")
    parser.add_argument("--seed",type=int,default=42)
    parser.add_argument("--project_desc",type=str, default = "Fine tuning llm")
    parser.add_argument("--name",type=str,default=None, help="file name to add")
    parser.add_argument("--full_ft",action="store_true",help="full finetuning otherwise lora")
    return parser.parse_args()


def main(args):
    seed_everything(args.seed)

    ## load tokenizer and model
    if args.name == "lora_sftt":
        model, tokenizer=load_model_tokenizer(args.base_model)
        model.resize_token_embeddings(len(tokenizer))
        model.config.use_cache = False
  

    ## dataset
    if args.name == "lora_sftt":
        train_dataset = load_and_prepare_dataset(source_dataset(args.dataset_path), args.name, "train")
        valid_dataset = load_and_prepare_dataset(source_dataset(args.dataset_path), args.name, "validation")
        test_dataset = load_and_prepare_dataset(source_dataset(args.dataset_path), args.name, "test")

        train_dataset=tokenize_dataset(train_dataset,tokenizer,args.max_len)
        valid_dataset=tokenize_dataset(valid_dataset,tokenizer,args.max_len)
        test_dataset=tokenize_dataset(test_dataset,tokenizer,args.max_len)

    response_template="###output:"
    data_collator=DataCollatorForCompletionOnlyLM(tokenizer.encode(response_template,add_special_tokens=False),tokenizer=tokenizer)
    
    ## wandb
    wandb.login(key = args.wandb_key)
    run = wandb.init(project=args.project_desc, job_type="training", anonymous="allow")

    if not args.full_ft:
        ## peft (lora)
        model = FastLanguageModel.get_peft_model(
            model,
            r = args.lora_r,
            target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj",],
            lora_alpha = args.lora_alpha,
            lora_dropout = args.dropout_rate, # Supports any, but = 0 is optimized
            bias = "none",    # Supports any, but = "none" is optimized
            use_gradient_checkpointing = True if args.gradient_checkpointing else False,
            random_state = args.seed,
            max_seq_length = 4096,
            use_rslora = False,  # We support rank stabilized LoRA
            loftq_config = None, # And LoftQ
        )

    if args.name is not None:
        output_dir= f"./checkpoints/{args.base_model.split('/')[-1]}_{args.name}"
    else:
        output_dir= f"./checkpoints/{args.base_model.split('/')[-1]}"
    
    time_now = datetime.today().strftime('%m%d%H%M')
    total_update_steps=int((len(train_dataset)*args.epochs)/(args.batch_size*args.gradient_accumulation_steps))
    eval_steps=int(total_update_steps/(args.epochs*args.num_save_per_epoch))
    
    training_args = TrainingArguments(
        output_dir = output_dir,
        num_train_epochs=args.epochs,
        evaluation_strategy='steps',
        metric_for_best_model='eval_loss',
        # per_device_train_batch_size=4,
        gradient_accumulation_steps=2,
        gradient_checkpointing=True if args.gradient_checkpointing else False,
        bf16=False, # bf16 is not supported by non-Ampere GPUs
        fp16=True,
        tf32=False,
        group_by_length=True, # pad batches by its group, more efficient
        load_best_model_at_end=True,
        report_to = 'wandb',
        # disable_tqdm=False,  # disable tqdm since with packing values are in correct
    )

    training_args = training_args.set_dataloader(train_batch_size=args.batch_size,
                                                 eval_batch_size=args.batch_size,
                                                 pin_memory=True,
                                                 num_workers=4,
                                                 sampler_seed=args.seed)
    training_args = training_args.set_lr_scheduler(name='cosine', num_epochs=args.epochs, warmup_ratio=args.warmup_ratio,)
    training_args = training_args.set_optimizer(name='paged_adamw_8bit', learning_rate=args.learning_rate, weight_decay=args.weight_decay,)
    training_args = training_args.set_evaluate(strategy = 'steps', steps = eval_steps, delay = 0, batch_size = args.batch_size)
    training_args = training_args.set_save(strategy="steps", steps = eval_steps, total_limit=10)
    training_args = training_args.set_logging(strategy="steps", steps=eval_steps, report_to = ['wandb'])
       
    metric = load_metric("sacrebleu")

    def compute_metrics(eval_preds):
        preds, labels = eval_preds 
        # In case the model returns more than the prediction logits 
        if isinstance(preds, tuple):
            preds = preds[0] 
        
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        
        # Replace -100s in the labels as we can't decode them 
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id) 
        decoded_labels = tokenizer.batch_decode(labels, skip_spacial_tokens=True) 
        
        # Some simple post-processing 
        decoded_preds = [pred.strip() for pred in decoded_preds]
        decoded_labels = [[label.strip()] for label in decoded_labels]
        
        result = metric.compute(predictions=decoded_preds, references=decoded_labels) 
        return {"bleu": result["score"]}

    trainer = SFTTrainer(
       compute_metrics=compute_metrics,
        model=model,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        max_seq_length= args.max_len,
        dataset_text_field="text",
        data_collator=data_collator,
        tokenizer=tokenizer,
        args=training_args,
        # neftune_noise_alpha=5,
        # packing= True,
        )
    
    if args.train:
        if args.ckpt_path is not None:
            trainer.train(args.ckpt_path)
        else:
            trainer.train()
            trainer.save_model()
                
    if args.test:
        metrics=trainer.evaluate(eval_dataset=test_dataset)
        print(f"test emtrics : {metrics}")

if __name__=="__main__":
  args=parse_args()
  main(args)