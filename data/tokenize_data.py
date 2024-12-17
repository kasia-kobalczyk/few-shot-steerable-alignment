from transformers import AutoTokenizer
from datasets import load_from_disk, Dataset
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--llm_name', type=str, required=True)
parser.add_argument('--max_length', type=int, required=True)

args = parser.parse_args()

llm_name = args.llm_name
max_length = args.max_length

prompt_template = "Human: {prompt}\\n\\nAssistant: "
template = prompt_template + "{response}"

tokenizer = AutoTokenizer.from_pretrained(llm_name)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"


def _tokenize(prompt, response):
    option = template.format(prompt=prompt, response=response)
    tokens = tokenizer(option)
    length = len(tokens["input_ids"])
    if length >= max_length:
        return None
    else:
        tokens = tokenizer(option, max_length=max_length, padding='max_length', truncation=True)
    assert len(tokens["input_ids"]) == max_length
    assert tokens["input_ids"][-1] == tokenizer.eos_token_id or tokens["input_ids"][-1] == tokenizer.pad_token_id
    cls_idx = length
    # get start index of response
    prompt_tokens = tokenizer(prompt_template.format(prompt=prompt))
    response_start_idx = len(prompt_tokens["input_ids"])
    return {
        "input_ids": tokens["input_ids"],
        "attention_mask": tokens["attention_mask"],
        "cls_idx": cls_idx,
        "response_start_idx": response_start_idx,
    }


def tokenize(example):
    tokens_a = _tokenize(example["prompt"], example["response_a"])
    tokens_b = _tokenize(example["prompt"], example["response_b"])

    if tokens_a is None or tokens_b is None:
        return {
            'option_a': None,
            'option_b': None,
        }
    else:
        return {
            'option_a': tokens_a,
            'option_b': tokens_b,
        }
    

dataset = load_from_disk("./data/ultra_feedback/raw_pairs")
tokenized_dataset = dataset.map(tokenize, batched=False)
tokenized_dataset = tokenized_dataset.to_pandas()
tokenized_dataset.dropna(inplace=True, subset=['option_a', 'option_b']) # drop pairs that are too long
tokenized_dataset = Dataset.from_pandas(tokenized_dataset)
tokenized_dataset.save_to_disk(f"./data/ultra_feedback/tokenized_pairs_{max_length}/{llm_name}")