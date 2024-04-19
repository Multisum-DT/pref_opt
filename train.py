import argparse
import gc
import re
import os
import random
import numpy as np
import torch
import wandb
import datasets
import evaluate
from datetime import datetime

from datasets import load_dataset, load_dataset_builder, Dataset, VerificationMode
from peft import (
    LoraConfig, 
    PeftModel, 
    prepare_model_for_kbit_training, 
    get_peft_model
)
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer
)
from trl import SFTTrainer, DPOTrainer, CPOTrainer, ORPOTrainer, CPOConfig
from unsloth import FastLanguageModel
from datasets import load_metric
from utils import *

os.environ["WANDB_API_KEY"] = 'e0079cf04794e1722592862727127f5711144304'
MODEL_MAPPER = {'mistral': 'mistralai/Mistral-7B-Instruct-v0.2', 
                'llama': 'meta-llama/Llama-2-7b-chat-hf', 
                'gemma': 'google/gemma-7b', 
                'bloomz7b': 'bigscience/bloomz-7b1',
                'bloomz1b': 'bigscience/bloomz-1b7',
                'opt': 'facebook/opt-1.3b',
                'tinyllama': 'TinyLlama/TinyLlama-1.1B-Chat-v1.0',
                }
CHAT_TEMPLATE_MAPPER = {'mistral': apply_chat_template_mistral,
                        'llama': apply_chat_template_llama, 
                        'gemma': None, 
                        'bloomz7b': None,
                        'bloomz1b': None,
                        'opt': None,
                        'tinyllama': None,
                        }


def compute_metrics(eval_preds):
    def split_input_output(string):
        input_label = string.split('### Input: ')[-1]
        input_str, label_str = input_label.split('### Output: ')
        return input_str.strip(), label_str.strip()
    
    preds, labels = eval_preds.predictions, eval_preds.label_ids
    # In case the model returns more than the prediction logits 
    if isinstance(preds, tuple):
        preds = preds[0]
    preds = preds.argmax(-1)
    preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    decoded_preds = list(map(lambda x: x.split('### Output: ')[-1].strip(), decoded_preds))
    
    # Replace -100s in the labels as we can't decode them
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_spacial_tokens=True)
    decoded_labels = [re.sub('<.*>', '', label).strip() for label in decoded_labels]
    decoded_inputs, decoded_labels = zip(*list(map(split_input_output, decoded_labels)))
    
    # Some simple post-processing 
    decoded_preds = [pred.strip() for pred in decoded_preds]
    decoded_labels = [label.strip() for label in decoded_labels]
    decoded_inputs = [input_id.strip() for input_id in decoded_inputs]
    
    # Multiple metrics
    metric = evaluate.combine(['sacrebleu', 'chrf'])
    metric_result = metric.compute(predictions = decoded_preds, references = decoded_labels)

    # Comet requires sources argument, make separate calculation
    comet = evaluate.load('comet')
    metric_result['comet'] = comet.compute(predictions = decoded_preds, references = decoded_labels, sources = decoded_inputs)['mean_score']

    return metric_result

def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)  # type: ignore
    torch.backends.cudnn.deterministic = True  # type: ignore
    torch.backends.cudnn.benchmark = True  # type: ignore

def parse_args():
    """
    model
    train
    lora_r
    lora_alpha
    lora_dropout
    epochs
    warmup_ratio
    learning_rate
    weight_decay
    num_save_per_epoch
    gradient_accumulation_steps
    """
    parser = argparse.ArgumentParser()
    
    ## directories
    parser.add_argument("--model",type=str, choices = ['mistral', 'llama', 'gemma', 'bloomz7b', 'bloomz1b', 'opt', 'tinyllama'],
                        default="mistral",help="Name of the model to be used")
    parser.add_argument("--train_mode", type=str, choices = ['sft', 'dpo', 'cpo', 'orpo'],
                        default = 'sft', help="Type of training to be used")
    parser.add_argument("--ckpt_dir",type=str,default=None)
    parser.add_argument("--level", type = str, default = 'sentence')
    # parser.add_argument("--mlflow_dir",type=str, default="mlruns")

    ## hyper parameters
    parser.add_argument("--seed",type=int,default=42)
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--max_len', type=int, default=4096)

    parser.add_argument("--lora_r", type=int, default=32)
    parser.add_argument("--lora_alpha", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.1)
    
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument("--warmup_ratio",type=float,default=0.1)
    parser.add_argument("--learning_rate", type=float, default=4e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--num_save_per_epoch",type=int,default=3,help="number of saving(evaluating) per a epoch")
    parser.add_argument("--gradient_accumulation_steps",type=int,default=1)
    parser.add_argument("--gradient_checkpointing", action='store_true',help="reduces required memory size but slows training")
    parser.add_argument("--eval_accumulation_steps", type=int, default = 0, help="reduces required memory size but slows training")
    
    return parser.parse_args()

def get_tok_and_model(model_path):
    if args.model in ['mistral', 'llama', 'gemma', 'tinyllama']:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_path,
            max_seq_length = 4096,
            dtype = torch.float16,
            load_in_4bit = True,
            cache_dir = '/data2/brian/.cache'
        )
        model = FastLanguageModel.get_peft_model(
            model,
            r = args.lora_r,
            target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj",],
            lora_alpha = args.lora_alpha,
            lora_dropout = args.dropout, # Supports any, but = 0 is optimized
            bias = "none",    # Supports any, but = "none" is optimized
            use_gradient_checkpointing = True if args.gradient_checkpointing else False,
            random_state = args.seed,
            max_seq_length = 4096,
            use_rslora = False,  # We support rank stabilized LoRA
            loftq_config = None, # And LoftQ
        )
    else:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            model_path, 
            cache_dir = '/data2/brian/.cache'
            )
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            # torch_dtype = torch.float16,
            device_map = 'auto',
            trust_remote_code = True,
            max_length = 4096,
            quantization_config = bnb_config,
            cache_dir = '/data2/brian/.cache',
        )
        peft_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.dropout,
            bias = 'none',
            task_type = 'CAUSAL_LM',
            target_modules = ['q_proj', 'k_proj', 'v_proj', 'o_proj', 
                              'gate_proj', 'up_proj', 'down_proj']
        )
        
        model.config.use_cache = False # use_cache is only for infernce
        model.enable_input_require_grads()
        model.config.pretraining_tp=1
        if args.gradient_checkpointing:
            model.gradient_checkpointing_enable()
        model = prepare_model_for_kbit_training(model)
        model = get_peft_model(model, peft_config)
    
    return model, tokenizer

def get_trainer(tokenizer, model, args):
    if args.train_mode == 'sft':
        builder = load_dataset_builder(
            './wmt14/wmt_utils.py',
            language_pair = ('fr', 'en'),
            subsets = {
                datasets.Split.TRAIN: ["europarl_v7", "newscommentary_v9"],
                datasets.Split.VALIDATION: ['newstest2013'],
                datasets.Split.TEST: ['newstest2014']
            },
            cache_dir = '/data2/brian/.cache/dataset'
        )
        builder.download_and_prepare(verification_mode=VerificationMode.NO_CHECKS)
        dataset = builder.as_dataset()
        train_dataset = dataset['train']
        eval_dataset = dataset['validation']
        test_dataset = dataset['test']
        del builder, dataset
        gc.collect()
    else:
        pass # TODO: ADD PO DATASET
        
    # create trainer for peft model
    time_now = datetime.today().strftime('%m%d%H%M')
    total_update_steps=int((len(train_dataset)*args.epochs)/(args.batch_size*args.gradient_accumulation_steps))
    eval_steps=int(total_update_steps/(args.epochs*args.num_save_per_epoch))

    output_dir = f'checkpoints/{args.model}_{time_now}'

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
        include_inputs_for_metrics=True,
        # disable_tqdm=False,  # disable tqdm since with packing values are in correct
    )

    training_args = training_args.set_dataloader(train_batch_size=args.batch_size,
                                                 eval_batch_size=args.batch_size,
                                                 pin_memory=True,
                                                 num_workers=4,
                                                 sampler_seed=args.seed)
    training_args = training_args.set_lr_scheduler(name='cosine', num_epochs=args.epochs, warmup_ratio=args.warmup_ratio,)
    training_args = training_args.set_optimizer(name='paged_adamw_8bit', learning_rate=args.learning_rate, weight_decay=args.weight_decay,)
    training_args = training_args.set_evaluate(strategy = 'steps', steps = eval_steps, delay = 0, accumulation_steps=args.eval_accumulation_steps, batch_size = args.batch_size)
    training_args = training_args.set_save(strategy="steps", steps = eval_steps, total_limit=10)
    training_args = training_args.set_logging(strategy="steps", steps=eval_steps, report_to = ['wandb'])
    
    wandb.init(
        # set the wandb project where this run will be logged
        project="translation-test",
        
        # track hyperparameters and run metadata
        config=training_args.__dict__
    )
    wandb.run.name = f"{args.model}_{time_now}"
    
    # train
    if args.train_mode == 'dpo':
        trainer = DPOTrainer(
            model=model,
            ref_model = None,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=compute_metrics,
            # peft_config=peft_config,
            max_seq_length=args.max_len,
            tokenizer=tokenizer,
            # packing=True,
            # dataset_text_field='text',
            args = training_args,
            formatting_func=CHAT_TEMPLATE_MAPPER[args.model], 
        )
    elif args.train_mode == 'cpo':
        trainer = CPOTrainer(
            
        )
    elif args.train_mode == 'orpo':
        trainer = ORPOTrainer(
            model=model,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=compute_metrics,
            # peft_config=peft_config,
            max_seq_length=args.max_len,
            tokenizer=tokenizer,
            # packing=True,
            # dataset_text_field='text',
            args = training_args,
            formatting_func=CHAT_TEMPLATE_MAPPER[args.model],
        )
    else:
        trainer = SFTTrainer(
            model=model,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=compute_metrics,
            # peft_config=peft_config,
            max_seq_length=args.max_len,
            tokenizer=tokenizer,
            # packing=True,
            # dataset_text_field='text',
            args = training_args,
            formatting_func=CHAT_TEMPLATE_MAPPER[args.model], 
        )    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    return trainer

if __name__ == '__main__':
    args = parse_args()
    seed_everything(args.seed)
    
    # Model Loading depending on name
    model_path = MODEL_MAPPER[args.model]
    model, tokenizer = get_tok_and_model(model_path)
    tokenizer.pad_token = tokenizer.unk_token

    # if args.use_flash_attn:
    #     from utils.llama_patch import upcast_layer_for_flash_attention
    #     torch_dtype = torch.bfloat16 if args.bf16 else torch.float16 if args.fp16 else torch.float32
    #     model = upcast_layer_for_flash_attention(model, torch_dtype)

    # get trainer
    trainer = get_trainer(tokenizer, model, args)
    

    if args.ckpt_dir:
        trainer.train(args.ckpt_dir)
    else:
        trainer.train()

    trainer.save_model(f'./final')