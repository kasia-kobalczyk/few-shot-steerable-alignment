import argparse
import random
import torch
from datasets import load_dataset, Dataset, DatasetDict, load_from_disk
import numpy as np
import os
from tqdm import tqdm 
import pandas as pd

seed=42
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)


llm_name = 'meta-llama/Meta-Llama-3-8B'
ds = load_from_disk(f'./data/ultra_feedback/embedded_pairs/{llm_name}')
df = ds.to_pandas().sort_values('idx').drop('__index_level_0__', axis=1)

def get_reward(scores, label='helpfulness'):
    if label == 'helpfulness':
        return scores[0]
    elif label == 'honesty':
        return scores[1]
    elif label == 'truthfulness':
        return scores[2]
    elif label == 'following_instructions':
        return scores[3]
    else:
        return None

def get_choice(reward_a, reward_b):
    if reward_a > reward_b:
        return 0.0
    elif reward_a < reward_b:
        return 1.0
    else:
        return 0.5

labels = ['helpfulness', 'honesty', 'truthfulness', 'following_instructions']

for label in labels:
    df[f'reward_a_{label}'] = df['scores_a'].apply(lambda x: get_reward(x, label))
    df[f'reward_b_{label}'] = df['scores_b'].apply(lambda x: get_reward(x, label))
    df[f'choice_{label}'] = df.apply(lambda x: get_choice(x[f'reward_a_{label}'], x[f'reward_b_{label}']), axis=1)


# Prepare the HH dataset
key_cols = ['idx', 'prompt', 'response_a', 'response_b']
labels = ['helpfulness', 'honesty']
hh_df = df[
    key_cols + 
    [f'choice_{label}' for label in labels] + 
    [f'reward_a_{label}' for label in labels] + 
    [f'reward_b_{label}' for label in labels]
].copy()

hh_df = hh_df[hh_df['choice_helpfulness'] != 0.5]
hh_df = hh_df[hh_df['choice_honesty'] != 0.5]

hh_df['is_conflicting'] = hh_df[f'choice_{labels[0]}'] != hh_df[f'choice_{labels[1]}']

n = len(hh_df[hh_df['is_conflicting']])
frac_conflicting = 1.0

conflicting_df = hh_df[hh_df['is_conflicting']].sample(int(n * frac_conflicting), random_state=seed)
non_conflicting_df = hh_df[~hh_df['is_conflicting']].sample(int(n * (1 - frac_conflicting)), random_state=seed)
sampled_df = pd.concat([conflicting_df, non_conflicting_df]).sample(frac=1, random_state=seed)
sampled_df['split'] = np.random.choice(['train', 'val', 'test'], size=len(sampled_df), p=[0.8, 0.1, 0.1])
sampled_df.to_csv(f'./data/ultra_feedback/hh_pairs_conflict_{frac_conflicting}.csv', index=False)


# Prepare the HHT dataset
key_cols = ['idx', 'prompt', 'response_a', 'response_b']
labels = ['helpfulness', 'honesty', 'truthfulness']
hht_df = df[
    key_cols + 
    [f'choice_{label}' for label in labels] + 
    [f'reward_a_{label}' for label in labels] + 
    [f'reward_b_{label}' for label in labels]
].copy()

# Remove pairs with ambiguous choices
hht_df = hht_df[hht_df['choice_helpfulness'] != 0.5]
hht_df = hht_df[hht_df['choice_honesty'] != 0.5]
hht_df = hht_df[hht_df['choice_truthfulness'] != 0.5]

sum_choices = hht_df[['choice_helpfulness', 'choice_honesty', 'choice_truthfulness']].sum(axis=1) 
hht_df['is_conflicting'] = (sum_choices == 1) | (sum_choices == 2)

n = len(hht_df[hht_df['is_conflicting']])
frac_conflicting = 1.0

conflicting_df = hht_df[hht_df['is_conflicting']].sample(int(n * frac_conflicting), random_state=seed)
non_conflicting_df = hht_df[~hht_df['is_conflicting']].sample(int(n * (1 - frac_conflicting)), random_state=seed)
sampled_df = pd.concat([conflicting_df, non_conflicting_df]).sample(frac=1, random_state=seed)
sampled_df['split'] = np.random.choice(['train', 'val', 'test'], size=len(sampled_df), p=[0.8, 0.1, 0.1])
sampled_df.to_csv(f'./data/ultra_feedback/hht_pairs_conflict_{frac_conflicting}.csv', index=False)