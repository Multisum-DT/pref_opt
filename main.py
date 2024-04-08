import argparse
import gc
import os
import random
import numpy as np
import torch
import wandb
import datasets

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
from trl import SFTTrainer, RewardTrainer
from unsloth import FastLanguageModel

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
    parser.add_argument("--model",type=str,default="mistral",help="Name of the model to be used")
    parser.add_argument("--train", action = 'store_true',help="Train model or only evaluate")
    parser.add_argument("--ckpt_dir",type=str,default=None)
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
    # parser.add_argument("--full_ft",action="store_true",help="full finetuning otherwise lora")
    # parser.add_argument("--chat_template", type=str,required=True,help="jinja chat template")
    
    ## etc
    # parser.add_argument("--expr_name",type=str,default=None, help="experiment name",required=True)
    # parser.add_argument("--expr_desc",type=str,help = "description for experiment", default = None)
    
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    seed_everything(args.seed)
    
    if args.model == 'mistral':
        model_path = 'mistralai/Mistral-7B-v0.1'
    elif args.model == 'llama13b':
        model_path = 'meta-llama/Llama-2-13b-hf'
    else:
        model_path = 'meta-llama/Llama-2-7b-hf'
    
    # load dataset

    chat_template = None # TODO
    tokenizer = AutoTokenizer.from_pretrained(model_path, cache_dir = '/data2/brian/.cache')
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.chat_template = chat_template
        
    # load bitsandbytes if inference
    # if not args.train:
    #     bnb_config = BitsAndBytesConfig(
    #         load_in_4bit=True,
    #         bnb_4bit_quant_type="nf4",
    #         bnb_4bit_compute_dtype=torch.float16,
    #         bnb_4bit_use_double_quant=True,
    #     )
    # else:
    #     bnb_config = BitsAndBytesConfig(
    #         load_in_8bit=True,
    #     )
    
    # load model
    # model = AutoModelForCausalLM.from_pretrained(
    #     model_path,
    #     cache_dir = '/data2/brian/.cache',
    #     # torch_dtype = torch.float16,
    #     load_in_8bit = True,
    #     device_map = 'auto',
    #     trust_remote_code = True,
    #     max_length = 4096,
    # )
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_path,
        max_seq_length = 4096,
        dtype = torch.float16,
        load_in_4bit = True,
        cache_dir = '/data2/brian/.cache'
    )
    tokenizer.pad_token = tokenizer.eos_token

    # if args.use_flash_attn:
    #     from utils.llama_patch import upcast_layer_for_flash_attention
    #     torch_dtype = torch.bfloat16 if args.bf16 else torch.float16 if args.fp16 else torch.float32
    #     model = upcast_layer_for_flash_attention(model, torch_dtype)

    # create lora config and create peft model
    if args.train:
        # peft_config = LoraConfig(
        #     r=args.lora_r,
        #     lora_alpha=args.lora_alpha,
        #     lora_dropout=args.dropout,
        #     bias = 'none',
        #     task_type = 'CAUSAL_LM',
        #     target_modules = ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']
        # )
        
        # model.config.use_cache = False # use_cache is only for infernce
        # model.enable_input_require_grads()
        # model.config.pretraining_tp=1
        # if args.gradient_checkpointing:
        #     model.gradient_checkpointing_enable()
        # model = prepare_model_for_kbit_training(model)
        
        # model = get_peft_model(model, peft_config)

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

    # load dataset
    if args.level == 'sentence':
        builder = load_dataset_builder(
            './wmt14/wmt_utils.py',
            language_pair = ('fr', 'en'),
            subsets = {
                datasets.Split.TRAIN: ["newscommentary_v10"],
                datasets.Split.VALIDATION: ['newstest2013'],
                datasets.Split.TEST: ['newstest2014']
            },
            cache_dir = '/data2/brian/.cache/dataset'
        )
        builder.download_and_prepare(verification_mode=VerificationMode.NO_CHECKS)
        dataset = builder.as_dataset()
        train_dataset = Dataset.load_from_disk('/data2/brian/personal/translation/sent-data/hf') # TODO: only europarl
        eval_dataset = dataset['validation']
        test_dataset = dataset['test']
        del builder, dataset
        gc.collect()
    else:
        train_dataset = Dataset.load_from_disk('/data2/brian/personal/translation/doc-data/hf')
        # TODO: construct doc level eval and test dataset
        
    

    # create trainer for peft model
    time_now = datetime.today().strftime('%m%d%H%M')
    total_update_steps=int((len(train_dataset)*args.epochs)/(args.batch_size*args.gradient_accumulation_steps))
    eval_steps=int(total_update_steps/(args.epochs*args.num_save_per_epoch))
    # eval_steps=1

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
    
    os.environ["WANDB_API_KEY"] = 'e0079cf04794e1722592862727127f5711144304'
    wandb.init(
        # set the wandb project where this run will be logged
        project="dolly-test",
        
        # track hyperparameters and run metadata
        config=training_args.__dict__
    )
    wandb.run.name = f"{args.model}_{time_now}"
    
    # train
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        # peft_config=peft_config,
        max_seq_length=args.max_len,
        tokenizer=tokenizer,
        # packing=True,
        dataset_text_field='text',
        args = training_args,
        # formatting_func=format_instruction, 
    )
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    if args.ckpt_dir:
        trainer.train(args.ckpr_dir)
    else:
        trainer.train()

    trainer.save_model(f'./final')