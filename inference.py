from transformers import AutoTokenizer, AutoModelForCausalLM
import datasets
from datasets import load_dataset_builder, VerificationMode
import random

model_path = '/home/yuntaeyang_0629/taeyang_2024/WMT_2024/pref_opt/checkpoints/mistral_04242037/checkpoint-53235'
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path)

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

batch = dataset['test']['translation'][random.randint(0, len(dataset))]
sys_prompt = 'You are a translator. Translate the sentence in French to English. Do not continue writing with anything that is unrelated to the given sentence.'
inference_sample = f"<s>[INST] {sys_prompt} {batch['fr']} [/INST] "
#inference_sample = f"<s>[INST] <<SYS>>\n{sys_prompt}\n<</SYS>>\n\n{batch['fr']} [/INST] "

output = model.generate(**tokenizer(inference_sample, return_tensors = 'pt'), max_length = 256)
print(tokenizer.decode(output[0]))