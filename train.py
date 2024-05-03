import argparse
import gc
import re
import os
import random
import numpy as np
import pandas as pd
import torch
import wandb
import datasets
import evaluate
from datetime import datetime
from tqdm import tqdm

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
    GenerationConfig,
    TrainingArguments,
    Trainer,
)
from transformers.integrations import WandbCallback
from trl import (
    SFTTrainer,
    DPOTrainer,
    CPOTrainer,
    ORPOTrainer, 
    CPOConfig, 
    ORPOConfig,
    DataCollatorForCompletionOnlyLM,
)
from unsloth import FastLanguageModel
from utils import *

os.environ["WANDB_API_KEY"] = 'e0079cf04794e1722592862727127f5711144304'
MODEL_MAPPER = {'mistral': 'mistralai/Mistral-7B-Instruct-v0.2', 
                'llama2': 'meta-llama/Llama-2-7b-chat-hf', 
                'gemma': 'google/gemma-7b', 
                'bloomz7b': 'bigscience/bloomz-7b1',
                'bloomz1b': 'bigscience/bloomz-1b7',
                'tinyllama': 'TinyLlama/TinyLlama-1.1B-Chat-v1.0',
                'llama3': 'meta-llama/Meta-Llama-3-8B-Instruct',
                }
CHAT_TEMPLATE_MAPPER = {'mistral': apply_chat_template_mistral,
                        'llama2': apply_chat_template_llama, 
                        'gemma': apply_chat_template_mistral, 
                        'bloomz7b': apply_chat_template_mistral,
                        'bloomz1b': apply_chat_template_mistral,
                        'tinyllama': apply_chat_template_tinyllama,
                        'llama3': apply_chat_template_llama3,
                        }

def compute_metrics(eval_preds):
    def split_input_output(string):
        input_str, label_str = string.split('[/INST]')

        # Extract the original sentence from the input message to be used for COMET
        sys_prompt = 'You are a translator. Translate the sentence in French to English. Directly start translating without answering back. Do not continue writing with anything that is unrelated to the given sentence.'
        input_str = input_str.replace('<<SYS>>', '').replace('<</SYS>>', '').replace('<s><s> [INST] ', '').replace(sys_prompt, '').strip()
        
        # Remove any special tokens that are not removed by tokenizer
        label_str = re.sub('<.*>', '', label_str).strip()
        return input_str, label_str
    
    def split_input_output_tinyllama(string):
        input_str, label_str = string.split('</s>\n<|assistant|>\n')

        # Extract the original sentence from the input message to be used for COMET
        input_str = input_str.split('</s>\n<|user|>\n')[-1].strip()
        
        # Remove any special tokens that are not removed by tokenizer
        label_str = re.sub('<.*>', '', label_str).strip()
        return input_str, label_str
    
    def split_input_output_llama3(string):
        input_str, label_str = string.split('<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n')

        # Extract the original sentence from the input message to be used for COMET
        input_str = input_str.split('<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n')[-1].strip()
        
        # Remove any special tokens that are not removed by tokenizer
        label_str = re.sub('<.*>', '', label_str).strip()
        return input_str, label_str
    
    preds, labels = eval_preds.predictions, eval_preds.label_ids
    # In case the model returns more than the prediction logits 
    if isinstance(preds, tuple):
        preds = preds[0]
    preds = preds.argmax(-1)
    preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
    preds = np.where(preds != 30488, preds, tokenizer.pad_token_id)
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    if 'llama-3' in tokenizer.name_or_path:
        decoded_preds = list(map(lambda x: x.split('<|start_header_id|>assistant<|end_header_id|>\n\n')[-1].strip(), decoded_preds))    
    elif 'tinyllama' in tokenizer.name_or_path:
        decoded_preds = list(map(lambda x: x.split('</s>\n<|assistant|>\n')[-1].strip(), decoded_preds))    
    else:
        decoded_preds = list(map(lambda x: x.split('[/INST]')[-1].strip(), decoded_preds))
    
    # Replace -100s in the labels as we can't decode them
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_spacial_tokens=True)
    if 'llama-3' in tokenizer.name_or_path:
        decoded_inputs, decoded_labels = zip(*list(map(split_input_output_llama3, decoded_labels)))
    elif 'tinyllama' in tokenizer.name_or_path:
        decoded_inputs, decoded_labels = zip(*list(map(split_input_output_tinyllama, decoded_labels)))
    else:
        decoded_inputs, decoded_labels = zip(*list(map(split_input_output, decoded_labels)))

    # Some simple post-processing 
    decoded_preds = [pred.strip() for pred in decoded_preds]
    decoded_labels = [label.strip() for label in decoded_labels]
    decoded_inputs = [input_id.strip() for input_id in decoded_inputs]
    
    # Multiple metrics
    metric = evaluate.combine(['sacrebleu', 'chrf'])
    metric_result = metric.compute(predictions = decoded_preds, references = decoded_labels)
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
    parser.add_argument("--model",type=str, choices = ['mistral', 'llama2', 'gemma', 'bloomz7b', 'bloomz1b', 'opt', 'tinyllama', 'llama3'],
                        default="mistral",help="Name of the model to be used")
    parser.add_argument("--train_mode", type=str, choices = ['sft', 'dpo', 'cpo', 'orpo'],
                        default = 'sft', help="Type of training to be used")
    parser.add_argument("--ckpt_dir",type=str,default=None)
    parser.add_argument("--level", type = str, default = 'sentence')

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

def apply_po_template(model_name, df):
    sys_prompt = "You are a translator. Translate the sentence in French to English. Directly start translating without answering back. Do not continue writing with anything that is unrelated to the given sentence."
    templates = {
        'llama2': f"<s>[INST] <<SYS>>\n{sys_prompt}\n<</SYS>>\n\n%s [/INST]",
        'mistral': f"<s>[INST] {sys_prompt} %s [/INST]",
        'gemma': f"<s>[INST] {sys_prompt} %s [/INST]", 
        'bloomz7b': f"<s>[INST] {sys_prompt} %s [/INST]",
        'bloomz1b': f"<s>[INST] {sys_prompt} %s [/INST]",
        'tinyllama': f"<|system|>\n{sys_prompt}</s>\n<|user|>\n%s</s>",
        'llama3': f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{sys_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n%s<|eot_id|>",
    }
    
    eos_token = '<|eot_id|>' if model_name == 'llama3' else '</s>'
    df['prompt'] = df['prompt'].map(lambda x: VAL_TEMPLATE[model_name] % x)
    df['chosen'] = df['prompt'] + df['chosen'] + eos_token
    df['rejected'] = df['prompt'] + df['rejected'] + eos_token
    return Dataset.from_pandas(df)

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
            target_modules = ["query_key_value", "dense"] if 'bloomz' in args.model else ["q_proj", "k_proj", "v_proj", "o_proj"],
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
            target_modules = ["query_key_value", "dense"] if 'bloomz' in args.model else ["q_proj", "k_proj", "v_proj", "o_proj"],
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

        def insert_text(samples, template):
            import ipdb; ipdb.set_trace()
            samples['text'] = template(samples['translation'])
            return samples
        
        train_dataset = dataset['train'].map(insert_text, fn_kwargs = {'template': CHAT_TEMPLATE_MAPPER[args.model]})
        eval_dataset = dataset['validation'].map(insert_text, fn_kwargs = {'template': CHAT_TEMPLATE_MAPPER[args.model]})
        test_dataset = dataset['test'].map(insert_text, fn_kwargs = {'template': CHAT_TEMPLATE_MAPPER[args.model]})

        train_dataset=tokenize_dataset(train_dataset,tokenizer,args.max_len)
        valid_dataset=tokenize_dataset(valid_dataset,tokenizer,args.max_len)
        test_dataset=tokenize_dataset(test_dataset,tokenizer,args.max_len)
        
        del builder, dataset
        gc.collect()
    else:
        # REMOVE
        # dataset = load_dataset('argilla/ultrafeedback-binarized-preferences-cleaned', cache_dir='/data2/brian/.cache/dataset')['train'] # prompt, chosen, rejected
        # dataset = Dataset.from_dict({
        #     'prompt': dataset['prompt'],
        #     'chosen': [tokenizer.apply_chat_template(sample, tokenize = False) for sample in dataset['chosen']],
        #     'rejected': [tokenizer.apply_chat_template(sample, tokenize = False) for sample in dataset['rejected']],
        # })

        # TODO: add apply_chat_template func for translation dataset
        train_df = pd.read_csv('po_valid_processing.csv')
        eval_df = pd.read_csv('po_valid_processing.csv')
        # test_df = pd.read_csv('po_valid_processing.csv')

        train_dataset = apply_po_template(args.model, train_df)
        eval_dataset = apply_po_template(args.model, eval_df)
        # test_dataset = apply_po_template(args.model, test_df)
    
    if args.model == 'llama3':
        response_template = '<|start_header_id|>assistant<|end_header_id|>\n\n'
    elif args.model == 'tinyllama':
        response_template = '\n<|assistant|>\n'
    else:
        response_template = ''
    data_collator = DataCollatorForCompletionOnlyLM(tokenizer(response_template, add_special_tokens = False), tokenizer = tokenizer)

    # create trainer for peft model
    time_now = datetime.today().strftime('%m%d%H%M')
    total_update_steps=int((len(train_dataset)*args.epochs)/(args.batch_size*args.gradient_accumulation_steps))
    eval_steps=int(total_update_steps/(args.epochs*args.num_save_per_epoch))

    output_dir = f'checkpoints/{args.model}_{args.train_mode}_{time_now}'
    group_by_length = True if args.train_mode == 'sft' else False
    generate_during_eval = False # if args.train_mode == 'sft' else True
    
    if args.train_mode == 'cpo':
        training_args = CPOConfig(
            output_dir = output_dir,
            num_train_epochs=args.epochs,
            evaluation_strategy='steps',
            metric_for_best_model='eval_loss',
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            gradient_checkpointing=True if args.gradient_checkpointing else False,
            bf16=True, # bf16 is not supported by non-Ampere GPUs
            fp16=False,
            tf32=False,
            group_by_length=group_by_length, # pad batches by its group, more efficient
            load_best_model_at_end=True,
            report_to = 'wandb',
            generate_during_eval=generate_during_eval,
            # include_inputs_for_metrics=True,
        )
        training_args = training_args.set_dataloader(train_batch_size=args.batch_size,
                                                    eval_batch_size=args.batch_size,
                                                    pin_memory=True,
                                                    num_workers=0,
                                                    sampler_seed=args.seed,
                                                    auto_find_batch_size=True,)
        training_args = training_args.set_lr_scheduler(name='cosine', num_epochs=args.epochs, warmup_ratio=args.warmup_ratio,)
        training_args = training_args.set_optimizer(name='paged_adamw_8bit', learning_rate=args.learning_rate, weight_decay=args.weight_decay,)
        training_args = training_args.set_evaluate(strategy = 'steps', steps = eval_steps, delay = 0, accumulation_steps=args.eval_accumulation_steps, batch_size = args.batch_size)
        training_args = training_args.set_save(strategy="steps", steps = eval_steps, total_limit=10)
        training_args = training_args.set_logging(strategy="steps", steps=eval_steps, report_to = ['wandb'])
    elif args.train_mode == 'orpo':
        training_args = ORPOConfig(
            output_dir = output_dir,
            num_train_epochs=args.epochs,
            evaluation_strategy='steps',
            metric_for_best_model='eval_loss',
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            gradient_checkpointing=True if args.gradient_checkpointing else False,
            bf16=True, # bf16 is not supported by non-Ampere GPUs
            fp16=False,
            tf32=False,
            group_by_length=group_by_length, # pad batches by its group, more efficient
            load_best_model_at_end=True,
            report_to = 'wandb',
            generate_during_eval=generate_during_eval,
            # include_inputs_for_metrics=True,
        )
        training_args = training_args.set_dataloader(train_batch_size=args.batch_size,
                                                    eval_batch_size=args.batch_size,
                                                    pin_memory=True,
                                                    num_workers=0,
                                                    sampler_seed=args.seed,
                                                    auto_find_batch_size=True,)
        training_args = training_args.set_lr_scheduler(name='cosine', num_epochs=args.epochs, warmup_ratio=args.warmup_ratio,)
        training_args = training_args.set_optimizer(name='paged_adamw_8bit', learning_rate=args.learning_rate, weight_decay=args.weight_decay,)
        training_args = training_args.set_evaluate(strategy = 'steps', steps = eval_steps, delay = 0, accumulation_steps=args.eval_accumulation_steps, batch_size = args.batch_size)
        training_args = training_args.set_save(strategy="steps", steps = eval_steps, total_limit=10)
        training_args = training_args.set_logging(strategy="steps", steps=eval_steps, report_to = ['wandb'])
    else:
        training_args = TrainingArguments(
            output_dir = output_dir,
            num_train_epochs=args.epochs,
            evaluation_strategy='steps',
            metric_for_best_model='eval_loss',
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            gradient_checkpointing=True if args.gradient_checkpointing else False,
            bf16=True, # bf16 is not supported by non-Ampere GPUs
            fp16=False,
            tf32=False,
            group_by_length=group_by_length, # pad batches by its group, more efficient
            load_best_model_at_end=True,
            report_to = 'wandb',
            #generate_during_eval=generate_during_eval,
            # include_inputs_for_metrics=True,
            # disable_tqdm=False,  # disable tqdm since with packing values are incorrect
        )

        training_args = training_args.set_dataloader(train_batch_size=args.batch_size,
                                                    eval_batch_size=args.batch_size,
                                                    pin_memory=True,
                                                    num_workers=0,
                                                    sampler_seed=args.seed,
                                                    auto_find_batch_size=True,)
        training_args = training_args.set_lr_scheduler(name='cosine', num_epochs=args.epochs, warmup_ratio=args.warmup_ratio,)
        training_args = training_args.set_optimizer(name='paged_adamw_8bit', learning_rate=args.learning_rate, weight_decay=args.weight_decay,)
        training_args = training_args.set_evaluate(strategy = 'steps', steps = eval_steps, delay = 0, accumulation_steps=args.eval_accumulation_steps, batch_size = args.batch_size)
        training_args = training_args.set_save(strategy="steps", steps = eval_steps, total_limit=10)
        training_args = training_args.set_logging(strategy="steps", steps=eval_steps, report_to = ['wandb'])
    
    os.environ["WANDB_API_KEY"] = '049ae4ba0b1bba160b91fd5b0c2a5a33b55cedfe'
    wandb.init(
        # set the wandb project where this run will be logged
        project="translation-test",
        
        # track hyperparameters and run metadata
        config=training_args.__dict__
    )
    wandb.run.name = f"{args.model}_{args.train_mode}_{time_now}"
    
    # train
    if args.train_mode == 'dpo':
        trainer = DPOTrainer(
            model=model,
            ref_model=None,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=None,
            tokenizer=tokenizer,
            args = training_args,
            # formatting_func=CHAT_TEMPLATE_MAPPER[args.model], 
        )
    elif args.train_mode == 'cpo':
        trainer = CPOTrainer(
            model=model,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=None,
            max_length=args.max_len,
            max_prompt_length=1024,
            tokenizer=tokenizer,
            args=training_args,
            # formatting_func=CHAT_TEMPLATE_MAPPER[args.model],
        )
    elif args.train_mode == 'orpo':
        trainer = ORPOTrainer(
            model=model,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            # max_length=args.max_len,
            compute_metrics=None,
            tokenizer=tokenizer,
            args = training_args,
            # formatting_func=CHAT_TEMPLATE_MAPPER[args.model],
        )
    else:
        trainer = SFTTrainer(
            model=model,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=compute_metrics,
            max_seq_length=args.max_len,
            tokenizer=tokenizer,
            args = training_args,
            data_collator = data_collator,
            formatting_func=CHAT_TEMPLATE_MAPPER[args.model],
            dataset_text_field="text",
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
    if args.model == 'llama3':
        tokenizer.pad_token = tokenizer.eos_token
    else:
        tokenizer.pad_token = tokenizer.unk_token

    # get trainer
    trainer = get_trainer(tokenizer, model, args)

    if args.ckpt_dir:
        trainer.train(args.ckpt_dir)
    else:
        trainer.train()

    trainer.save_model(f'./final')