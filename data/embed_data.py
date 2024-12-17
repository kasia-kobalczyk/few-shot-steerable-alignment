from datasets import load_from_disk, DatasetDict, Dataset, concatenate_datasets
import torch
from transformers import AutoModel
import pandas as pd
from multiprocessing import Pool, set_start_method
set_start_method('spawn', force=True)

def get_embedding(option, model):
    cls_idxs = torch.stack([torch.tensor(y['cls_idx']) for y in option]).to(model.device)
    input_ids = torch.stack([torch.tensor(y['input_ids']) for y in option]).to(model.device)
    attention_mask = torch.stack([torch.tensor(y['attention_mask']) for y in option]).to(model.device)
    with torch.no_grad():
        last_hidden_state = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        ).hidden_states[-1]
        bs = last_hidden_state.shape[0]
        embed = torch.zeros((bs, last_hidden_state.shape[2]))
        for i in range(bs):
            embed[i, :] = last_hidden_state[i, cls_idxs[i], :]

    return embed


def embed(example, model):
    options_a_embed = get_embedding(example["option_a"], model)
    options_b_embed = get_embedding(example["option_b"], model)

    return {
        'option_a': options_a_embed,
        'option_b': options_b_embed,
    }

def embed_dataset(dataset, model):
    print('Embedding dataset on device', model.device)
    return dataset.map(lambda x: embed(x, model), batched=True, batch_size=8)


if __name__ == '__main__':
    llm_name = "meta-llama/Meta-Llama-3-8B"
    ds = load_from_disk(f"./data/ultra_feedback/tokenized_pairs/{llm_name}")

    n = len(ds)
    n_devices = 4 # Number of GPUs

    devices = [f'cuda:{i}' for i in range(n_devices)]
    datasets = [
        ds.select(list(range(i, n, n_devices)))
        for i in range(n_devices)
    ]

    def initialize_model(device):
        print('Initializing model on', device)
        model = AutoModel.from_pretrained(
            llm_name,
            torch_dtype=torch.float16
        ).to(device)
        model.eval()
        return model

    models = [initialize_model(device) for device in devices]
        

    with Pool(n_devices) as pool:
        embedded_datasets = pool.starmap(embed_dataset, [(dataset, model) for dataset, model in zip(datasets, models)])

    embedded_dataset = concatenate_datasets(embedded_datasets)

    embedded_dataset.save_to_disk(f"./data/ultra_feedback/embedded_pairs/{llm_name}")