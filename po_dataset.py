import random
import numpy as np
import torch
import pandas as pd
import os
import ipdb
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig

import datasets
from datasets import load_dataset, load_dataset_builder, Dataset, VerificationMode
from sklearn.model_selection import train_test_split

from tqdm import tqdm
def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)  # type: ignore
    torch.backends.cudnn.deterministic = True  # type: ignore
    torch.backends.cudnn.benchmark = False  # type: ignore

if __name__ == '__main__':
    seed_everything(42)
    builder = load_dataset_builder(
        './wmt14/wmt_utils.py',
        language_pair = ('fr', 'en'),
        subsets = {
            datasets.Split.TRAIN: ["europarl_v7", "newscommentary_v9"],
            datasets.Split.VALIDATION: ['newstest2013'],
            datasets.Split.TEST: ['newstest2014']
            },
            #cache_dir = '/data2/brian/.cache/dataset'
            )
    builder.download_and_prepare(verification_mode=VerificationMode.NO_CHECKS)
    dataset = builder.as_dataset()

    _, train_dataset = dataset['train'].train_test_split(test_size=0.1).values()
    gemma_train, llama_mistral_train = train_dataset.train_test_split(test_size=0.6).values()
    llama_train, mistral_train = llama_mistral_train.train_test_split(test_size=0.3).values()

    eval_dataset = dataset['validation']
    gemma_eval, llama_mistral_eval = eval_dataset.train_test_split(test_size=0.6).values()
    llama_eval, mistral_eval = llama_mistral_eval.train_test_split(test_size=0.3).values()    

    test_dataset = dataset['test']
    gemma_test, llama_mistral_test = test_dataset.train_test_split(test_size=0.6).values()
    llama_test, mistral_test = llama_mistral_test.train_test_split(test_size=0.3).values()

    base_model_name1="google/gemma-1.1-7b-it"
    gemma_tokenizer = AutoTokenizer.from_pretrained(base_model_name1)
    gemma_model= AutoModelForCausalLM.from_pretrained(base_model_name1, device_map="auto",torch_dtype=torch.float16)

    base_model_name2="meta-llama/Llama-2-7b-chat-hf"
    llama_tokenizer = AutoTokenizer.from_pretrained(base_model_name2)
    llama_model= AutoModelForCausalLM.from_pretrained(base_model_name2, device_map="auto",torch_dtype=torch.float16)

    base_model_name3="mistralai/Mistral-7B-Instruct-v0.2"
    mistral_tokenizer = AutoTokenizer.from_pretrained(base_model_name3)
    mistral_model= AutoModelForCausalLM.from_pretrained(base_model_name3, device_map="auto",torch_dtype=torch.float16)
    def replace_llama(data_type, sys_prompt, data, idx):
        data_type.loc[idx, "prompt"] = f"<s>[INST] <<SYS>>\n{sys_prompt}\n<</SYS>>\n\n{data['fr']} [/INST]"

        input_ids = llama_tokenizer(data_type.loc[idx, "prompt"], return_tensors="pt").to("cuda")
        gened = llama_model.generate(**input_ids, max_new_tokens=200)
        result_str = llama_tokenizer.decode(gened[0])
        start_tag = f"[/INST]"
        start_index = result_str.find(start_tag)
        if start_index != -1:
            result_str = result_str[start_index + len(start_tag):].strip()
        data_type.loc[idx, "chosen"] = data['en']
        data_type.loc[idx, "rejected"] = result_str

    def get_data(data_type, gemma_data, llama_data, mistral_data):
        idx = 0
        sys_prompt = 'You are a translator. Translate the sentence in French to English. Do not continue writing with anything that is unrelated to the given sentence.'
        for data in tqdm(gemma_data['translation']):
            data_type.loc[idx, "prompt"] = f"<s>[INST] {sys_prompt} {data['fr']} [/INST]"

            input_ids = gemma_tokenizer(data_type.loc[idx, "prompt"], return_tensors="pt").to("cuda")
            gened = gemma_model.generate(**input_ids, max_new_tokens=200)
            result_str = gemma_tokenizer.decode(gened[0])
            start_tag = f"[/INST]"
            start_index = result_str.find(start_tag)
            if start_index != -1:
                result_str = result_str[start_index + len(start_tag):].strip()
            data_type.loc[idx, "chosen"] = data['en']
            if "<pad>"*10 in result_str:
                replace_llama(data_type, sys_prompt, data, idx)
            else:
                data_type.loc[idx, "rejected"] = result_str
            idx +=1
        data_type.to_csv("po_train_gemma.csv", index=False)
        for data in tqdm(llama_data['translation']):
            data_type.loc[idx, "prompt"] = f"<s>[INST] <<SYS>>\n{sys_prompt}\n<</SYS>>\n\n{data['fr']} [/INST]"

            input_ids = llama_tokenizer(data_type.loc[idx, "prompt"], return_tensors="pt").to("cuda")
            gened = llama_model.generate(**input_ids, max_new_tokens=200)
            result_str = llama_tokenizer.decode(gened[0])
            start_tag = f"[/INST]"
            start_index = result_str.find(start_tag)
            if start_index != -1:
                result_str = result_str[start_index + len(start_tag):].strip()
            data_type.loc[idx, "chosen"] = data['en']
            data_type.loc[idx, "rejected"] = result_str
            idx += 1
        data_type.to_csv("po_train_gemma_llama.csv", index=False)
        for data in tqdm(mistral_data['translation']):
            data_type.loc[idx, "prompt"] = f"<s>[INST] {sys_prompt} {data['fr']} [/INST]"

            input_ids = mistral_tokenizer(data_type.loc[idx, "prompt"], return_tensors="pt").to("cuda")
            gened = mistral_model.generate(**input_ids, max_new_tokens=200)
            result_str = mistral_tokenizer.decode(gened[0])
            start_tag = f"[/INST]"
            start_index = result_str.find(start_tag)
            if start_index != -1:
                result_str = result_str[start_index + len(start_tag):].strip()
            data_type.loc[idx, "chosen"] = data['en']
            data_type.loc[idx, "rejected"] = result_str
            idx += 1

    po_train=pd.DataFrame(columns=["prompt", "chosen", "rejected"])
    #po_valid=pd.DataFrame(columns=["prompt", "chosen", "rejected"])
    #po_test=pd.DataFrame(columns=["prompt", "chosen", "rejected"])
    get_data(po_train, gemma_train, llama_train, mistral_train)
    po_train.to_csv("po_train.csv", index=False)
    #get_data(po_valid, gemma_eval, llama_eval, mistral_eval)
    #po_valid.to_csv("po_valid.csv", index=False)
    #get_data(po_test, gemma_test, llama_test, mistral_test)
    #po_test.to_csv("po_test.csv", index=False)

