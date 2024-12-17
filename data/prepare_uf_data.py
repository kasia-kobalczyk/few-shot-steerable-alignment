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

ultra_feedback = load_dataset('openbmb/UltraFeedback')

def get_scores(x):
    try:
        helpfulness = int(x['annotations']['helpfulness']['Rating'])
        honesty =  int(x['annotations']['honesty']['Rating'])
        truthfulness = int(x['annotations']['truthfulness']['Rating'])
        following_instructions = int(x['annotations']['instruction_following']['Rating'])
        return np.array([helpfulness, honesty, truthfulness, following_instructions])
    except:
        return None

def get_response(x):
    try:
        return x['response']
    except:
        return None
    
# Extract completions and scores
uf_df = ultra_feedback['train'].to_pandas().drop(['correct_answers', 'incorrect_answers', 'models'], axis=1)
uf_df = uf_df.explode('completions')
uf_df.reset_index(drop=True, inplace=True)
uf_df['scores'] = uf_df.completions.apply(lambda x: get_scores(x))
uf_df['response'] = uf_df.completions.apply(lambda x: get_response(x))
uf_df.dropna(inplace=True)
uf_df.drop('completions', axis=1, inplace=True)
uf_df.rename({'instruction': 'prompt'}, axis=1, inplace=True)

# Create a dataframe with all possible pairs of responses per instruction
def get_pairs(x):
    pairs = []
    for i in range(len(x)):
        for j in range(i+1, len(x)):
            pairs.append([x[i], x[j]])
    return pairs
response_pairs = uf_df.groupby('prompt')['response'].apply(lambda x: get_pairs(x.tolist())).reset_index()
score_pairs = uf_df.groupby('prompt')['scores'].apply(lambda x: get_pairs(x.tolist())).reset_index()
pairs = pd.merge(response_pairs, score_pairs, on='prompt')
pairs = pairs.explode(['response', 'scores']).dropna()

# Extract scores and responses
pairs['response_a'] = pairs['response'].apply(lambda x: x[0])
pairs['response_b'] = pairs['response'].apply(lambda x: x[1])
pairs['scores_a'] = pairs['scores'].apply(lambda x: x[0])
pairs['scores_b'] = pairs['scores'].apply(lambda x: x[1])
pairs.drop(['response', 'scores'], axis=1, inplace=True)
pairs.reset_index(drop=True, inplace=True)
pairs['idx'] = pairs.index

# Save pairs to disk
pairs_ds = Dataset.from_pandas(pairs)
pairs_ds.save_to_disk('./data/ultra_feedback/raw_pairs_test')