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

def apply_chat_template_llama(batch):
    sys_prompt = "You are a translator. Translate the sentence in French to English. Directly start translating without answering back. Do not continue writing with anything that is unrelated to the given sentence."
    batch = f"""<s>[INST] <<SYS>>\n{sys_prompt}\n<</SYS>>\n\n{batch['fr']} [/INST]{batch['en']}</s>"""
    return batch

def apply_chat_template_mistral(batch):
   sys_prompt = 'You are a translator. Translate the sentence in French to English. Directly start translating without answering back. Do not continue writing with anything that is unrelated to the given sentence.'
   batch= f"<s>[INST] {sys_prompt} {batch['fr']} [/INST] {batch['en']}</s>"
   return batch

def apply_chat_template_tinyllama(batch):
   sys_prompt = 'You are a translator. Translate the sentence in French to English. Directly start translating without answering back. Do not continue writing with anything that is unrelated to the given sentence.'
   batch = f"<|system|>\n{sys_prompt}</s>\n<|user|>\n{batch['fr']}</s>\n<|assistant|>\n{batch['en']}</s>"
   return batch

def apply_chat_template_llama3(batch):
   sys_prompt = 'You are a translator. Translate the sentence in French to English. Directly start translating without answering back. Do not continue writing with anything that is unrelated to the given sentence.'
   batch = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{sys_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{batch['fr']}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n{batch['en']}<|eot_id|>"
   return batch
   
# def apply_chat_template_llama(batch):
#     samples = []
#     sys_prompt = "You are a translator. Translate the sentence in French to English. Directly start translating without answering back. Do not continue writing with anything that is unrelated to the given sentence."
#     for i in range(len(batch['translation'])):
#        samples.append(f"""<s>[INST] <<SYS>>\n{sys_prompt}\n<</SYS>>\n\n{batch['translation']['fr']} [/INST]{batch['translation']['en']}</s>""")
#     return samples

# def apply_chat_template_mistral(batch):
#    sys_prompt = 'You are a translator. Translate the sentence in French to English. Directly start translating without answering back. Do not continue writing with anything that is unrelated to the given sentence.'
#    samples = []
#    for i in range(len(batch['translation'])):
#       samples.append(f"<s>[INST] {sys_prompt} {batch['translation'][i]['fr']} [/INST] {batch['translation'][i]['en']}</s>")
#    return samples

# def apply_chat_template_tinyllama(batch):
#    sys_prompt = 'You are a translator. Translate the sentence in French to English. Directly start translating without answering back. Do not continue writing with anything that is unrelated to the given sentence.'
#    samples = []
#    for i in range(len(batch['translation'])):
#       samples.append(f"<|system|>\n{sys_prompt}</s>\n<|user|>\n{batch['translation'][i]['fr']}</s>\n<|assistant|>\n{batch['translation'][i]['en']}</s>")
#    return samples

# def apply_chat_template_llama3(batch):
#    sys_prompt = 'You are a translator. Translate the sentence in French to English. Directly start translating without answering back. Do not continue writing with anything that is unrelated to the given sentence.'
#    samples = []
#    for i in range(len(batch['translation'])):
#       samples.append(f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{sys_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{batch['translation'][i]['fr']}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n{batch['translation'][i]['en']}<|eot_id|>")
#    return samples

def apply_chat_template_llama_po(batch):
    samples = []
    system_message = "You are a translator. Translate the sentence in French to English. Directly start translating without answering back. Do not continue writing with anything that is unrelated to the given sentence."
    for i in range(len(batch)):
      samples.append({'prompt': batch[i]['prompt'],
                      'chosen': batch[i]['chosen'],
                      'rejected': batch[i]['rejected']})
      #  samples.append(f"""<s>[INST] <<SYS>>\n{system_message}\n<</SYS>>\n\n{batch['translation'][i]['fr']} [/INST]{batch['translation'][i]['en']}</s>""")
    return samples

def apply_chat_template_mistral_po(batch):
   sys_prompt = 'You are a translator. Translate the sentence in French to English. Directly start translating without answering back. Do not continue writing with anything that is unrelated to the given sentence.'
   samples = []
   for i in range(len(batch['translation'])):
      samples.append({'prompt': batch[i]['prompt'],
                      'chosen': batch[i]['chosen'],
                      'rejected': batch[i]['rejected']})
      # samples.append(f"<s>[INST] {sys_prompt} {batch['translation'][i]['fr']} [/INST] {batch['translation'][i]['en']}</s>")
   return samples

def apply_chat_template_tinyllama_po(batch):
   sys_prompt = 'You are a translator. Translate the sentence in French to English. Directly start translating without answering back. Do not continue writing with anything that is unrelated to the given sentence.'
   samples = []
   for i in range(len(batch['translation'])):
      samples.append({'prompt': batch[i]['prompt'],
                      'chosen': batch[i]['chosen'],
                      'rejected': batch[i]['rejected']})
      # samples.append(f"<|system|>\n{sys_prompt}</s>\n<|user|>\n{batch['translation'][i]['fr']}</s>\n<|assistant|>\n{batch['translation'][i]['en']}</s>")
   return samples

def apply_chat_template_llama3_po(batch):
   sys_prompt = 'You are a translator. Translate the sentence in French to English. Directly start translating without answering back. Do not continue writing with anything that is unrelated to the given sentence.'
   samples = []
   for i in range(len(batch['translation'])):
      samples.append({'prompt': batch[i]['prompt'],
                      'chosen': batch[i]['chosen'],
                      'rejected': batch[i]['rejected']})
      # samples.append(f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{sys_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{batch['translation'][i]['fr']}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n{batch['translation'][i]['en']}<|eot_id|>")
   return samples

def return_prompt_and_responses(batch):
    # samples['text']="###input: " + samples['translation']['fr'] + "###instruction: Please translate the input French sentence into English" + "###output:" + samples['translation']['en']
    samples = []
    for i in range(len(batch['translation'])):
       samples.append(f"<s>### Instruction: Please translate the input sentence written in French to English\n### Input: {batch['translation'][i]['fr']}\n### Output: {batch['translation'][i]['en']}<\s>")
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