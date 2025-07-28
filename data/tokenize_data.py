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
bos_token = tokenizer.bos_token if tokenizer.bos_token is not None else ""
system_prompt = "You are a helpful AI assistant."

def _get_chat_text(prompt, response):
    if hasattr(tokenizer, 'apply_chat_template'):
        # Use built-in chat template if available
        messages = [
            #{"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response}
        ]
        return tokenizer.apply_chat_template(messages, tokenize=False)
    else:
        # Fallback to manual template
        print("Fallback to manual template")
        return bos_token + template.format(prompt=prompt, response=response)

def _tokenize(prompt, response):
    # Replace existing full_text construction with chat template
    full_text = _get_chat_text(prompt, response)
    tokens = tokenizer(full_text)
    length = len(tokens["input_ids"])
    if length >= max_length:
        return None
    
    # Get response start index using chat template
    prefix_text = _get_chat_text(prompt, "")
    prefix_tokens = tokenizer(prefix_text)
    response_start_idx = len(prefix_tokens["input_ids"])
    
    # Tokenize with padding
    tokens = tokenizer(
        full_text,
        max_length=max_length,
        padding='max_length',
        truncation=True,
        return_tensors=None
    )
    
    # Ensure EOS token is properly placed
    if tokens["input_ids"][-1] != tokenizer.eos_token_id:
        tokens["input_ids"][-1] = tokenizer.eos_token_id
    
    # Create position IDs (start from 0 for padding tokens)
    position_ids = [i if m == 1 else 0 for i, m in enumerate(tokens["attention_mask"])]
    
    return {
        "input_ids": tokens["input_ids"],
        "attention_mask": tokens["attention_mask"],
        "position_ids": position_ids,
        "cls_idx": length,  # Original sequence length before padding
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
    

dataset = load_from_disk("/mnt/pdata/knk25/cPL/data/ultra_feedback/raw_pairs")
tokenized_dataset = dataset.map(tokenize, batched=False)
tokenized_dataset = tokenized_dataset.to_pandas()
tokenized_dataset.dropna(inplace=True, subset=['option_a', 'option_b']) # drop pairs that are too long
tokenized_dataset = Dataset.from_pandas(tokenized_dataset)
tokenized_dataset.save_to_disk(f"/mnt/pdata/caf83/few-shot-alignment/data/ultra_feedback/tokenized_pairs_{max_length}/{llm_name}")