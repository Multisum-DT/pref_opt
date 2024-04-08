from transformers import AutoModelForCausalLM, AutoTokenizer
import datasets
from datasets import load_dataset_builder, VerificationMode
from unsloth import FastLanguageModel

import torch
import numpy as np
import random
import os
from typing import List

def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)  # type: ignore
    torch.backends.cudnn.deterministic = True  # type: ignore
    torch.backends.cudnn.benchmark = False  # type: ignore

def source_dataset(dataset_path):
    builder = load_dataset_builder(
        dataset_path,
        language_pair = ('fr', 'en'),
        subsets = {
            datasets.Split.TRAIN: ["europarl_v7", "newscommentary_v10"],
            datasets.Split.VALIDATION: ['newstest2013'],
            datasets.Split.TEST: ['newstest2014']
        },

    )

    builder.download_and_prepare(verification_mode=VerificationMode.NO_CHECKS)

    dataset = builder.as_dataset()

    return dataset

def return_prompt_and_responses(samples): 
    samples['text']="###input: " + samples['translation']['en'] + "###instruction: Please translate the input English sentence into French" + "###output:" + samples['translation']['fr']
    return samples

def return_prompt_and_responses_dpo(samples):
  samples['prompt'] = samples["translation"]['en']
  samples['chosen'] = samples["translation"]['fr']
  samples["rejected"] = samples["translation"]['fr']
  return samples

def load_and_prepare_dataset(source_dataset, train_types, data_types):
    if data_types=="train":
        dataset=source_dataset['train']
        original_columns = dataset.column_names
    elif data_types=="validation":
        dataset=source_dataset['validation']
        original_columns = dataset.column_names
    else:
        dataset=source_dataset['test']
        original_columns = dataset.column_names
        
    if train_types == "lora_sftt":
      dataset = dataset.map(
        return_prompt_and_responses
        )
      dataset = dataset.map( 
      batched=True,
      remove_columns=original_columns
      )
    elif train_types == "dpo":
      dataset = dataset.map(
        return_prompt_and_responses_dpo
        )
      dataset = dataset.map( 
      batched=True,
      remove_columns=original_columns
      )
    return dataset




def load_model_tokenizer(base_model_path, additional_special_tokens:List[str]=None):
    model, tokenizer = FastLanguageModel.from_pretrained(
        base_model_path,
        max_seq_length = 4096,
        dtype = torch.float16,
        load_in_4bit = True
    )

    if additional_special_tokens is not None:
      tokenizer.add_special_tokens(
        {"additional_special_tokens": additional_special_tokens}
      )

    tokenizer.padding_side="right"
    return model, tokenizer

def tokenize_dataset(dataset,tokenizer,max_length):
    # this fucntion guarantees every training samples must have complete input texts
      
  dataset=dataset.filter(lambda x: len(tokenizer.tokenize(x["text"]))<max_length-1) # should consider bos/eos token
  dataset=dataset.map(lambda x: tokenizer(x['text'],max_length=max_length), batched=True)
  dataset=dataset.shuffle()
  return dataset