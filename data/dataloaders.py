from torch.utils.data import Dataset
import torch 
import numpy as np
from datasets import load_from_disk
import os
import pandas as pd
import random

class PreferenceDataset(Dataset):
    def __init__(
            self, 
            path_to_data,
            split_file,
            split, 
            datatype='embeddings',
            labels=['helpfulness', 'honesty']
        ):
        self.data_index = pd.read_csv(split_file)
        self.data_index = self.data_index[self.data_index['split'] == split]
        self.labels = labels
        self.datatype = datatype

        self.data = load_from_disk(path_to_data).to_pandas()[['idx', 'option_a', 'option_b']]
        self.data = pd.merge(self.data_index, self.data, on='idx', how='left').set_index('idx')
        assert len(self.data) <= len(self.data_index)
        self.data = self.data.dropna(subset=['option_a', 'option_b'])

    def get(self, label, idx):
        data = self.data.loc[idx, :]
        if self.datatype != 'tokens':
            option_a = torch.stack([torch.tensor(x) for x in data['option_a'].values])
            option_b = torch.stack([torch.tensor(x) for x in data['option_b'].values])
            pairs = torch.stack([option_a, option_b], dim=1).float()
            if len(pairs.shape) == 2:
                pairs = pairs.unsqueeze(-1)
            return {
                'pairs': pairs,
                'choices': torch.tensor(data[f'choice_{label}'].values).reshape(-1, 1).long(),
                'label': label
            }
        elif self.datatype == 'tokens':
            option_a_input_ids = torch.stack([torch.tensor(x['input_ids']) for x in data['option_a'].values])
            option_a_attention_mask = torch.stack([torch.tensor(x['attention_mask']) for x in data['option_a'].values])
            option_b_input_ids = torch.stack([torch.tensor(x['input_ids']) for x in data['option_b'].values])
            option_b_attention_mask = torch.stack([torch.tensor(x['attention_mask']) for x in data['option_b'].values])
            option_a_cls_idxs = torch.tensor(data['option_a'].apply(lambda x: x['cls_idx']).values)
            option_b_cls_idxs = torch.tensor(data['option_b'].apply(lambda x: x['cls_idx']).values)
            option_a_response_start_idxs = torch.tensor(data['option_a'].apply(lambda x: x['response_start_idx']).values)
            option_b_response_start_idxs = torch.tensor(data['option_b'].apply(lambda x: x['response_start_idx']).values)
            pairs_input_ids = torch.stack([option_a_input_ids, option_b_input_ids], dim=1).long()
            pairs_attention_mask = torch.stack([option_a_attention_mask, option_b_attention_mask], dim=1).float()
            pairs_cls_idxs = torch.stack([option_a_cls_idxs, option_b_cls_idxs], dim=1).long()
            pairs_response_start_idxs = torch.stack([option_a_response_start_idxs, option_b_response_start_idxs], dim=1).long()
            return {
                'pairs_input_ids': pairs_input_ids,
                'pairs_attention_mask': pairs_attention_mask,
                'pairs_cls_idxs': pairs_cls_idxs,
                'pairs_response_start_idxs': pairs_response_start_idxs,
                'choices': torch.tensor(data[f'choice_{label}'].values).reshape(-1, 1).long(),
                'label': label
            }


class ContextTargetDataset(Dataset):
    def __init__(self, cfg, split, context_dataset, target_dataset):
        self.min_num_context = cfg.min_num_context
        self.max_num_context = cfg.max_num_context
        self.num_targets = cfg.num_targets
    
        self.context_dataset = context_dataset
        self.target_dataset = target_dataset
        self.labels = self.context_dataset.labels
       
        self.common_idx = sorted(list(set(context_dataset.data.index).intersection(set(target_dataset.data.index))))
        random.seed(42)  # Add this line to the __init__
        self.rng = random.Random(42)  # Create a separate Random instance

    def __getitem__(self, idx):
        label = self.rng.choice(self.labels)
        idx = self.rng.sample(self.common_idx, self.num_targets)

        target_data = self.target_dataset.get(label, idx)
        context_data = self.context_dataset.get(label, idx)

        return context_data, target_data
    
    def __len__(self):
        return len(self.common_idx)



class ConflictingDataset(Dataset):
    def __init__(self, num_targets, frac_context_conflict=0.5, frac_target_conflict=1.0, split='test'):
        
        self.context_dataset = PreferenceDataset(
            path_to_data=f'../data/ultra_feedback/embedded_pairs/meta-llama/Meta-Llama-3-8B',
            split_file=f'../data/ultra_feedback/hh_pairs_conflict_{frac_context_conflict}.csv',
            split=split,
            datatype='embeddings',
            labels=['helpfulness', 'honesty']
        )

        self.target_dataset = PreferenceDataset(
            path_to_data=f'../data/ultra_feedback/embedded_pairs/meta-llama/Meta-Llama-3-8B',
            split_file=f'../data/ultra_feedback/hh_pairs_conflict_{frac_target_conflict}.csv',
            split='test',
            datatype='embeddings',
            labels=['helpfulness', 'honesty']
        )

        self.num_targets = num_targets

    def __getitem__(self, idx):
        label = np.random.choice(self.context_dataset.labels)
        
        context_idx = list(self.context_dataset.data.sample(self.num_targets).index)

        target_idx = list(self.target_dataset.data.sample(self.num_targets).index)

        context_data = self.context_dataset.get(label, context_idx)
        target_data = self.target_dataset.get(label, target_idx)

        return context_data, target_data

    def __len__(self):
        return min(len(self.context_dataset.data), len(self.target_dataset.data))
