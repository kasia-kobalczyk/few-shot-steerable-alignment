from datasets import Dataset, DatasetDict
import numpy as np
import pandas as pd

def get_reward(y, z):
    thres = 0.5
    if z == 0:
        r = np.where(y < thres, y, 2*y)
    else:
        r = np.where(y < thres, y, 1 - y)
    return r

def get_choices(rewards_a, rewards_b):
    choices = np.zeros_like(rewards_a)
    choices[rewards_b > rewards_a] = 1
    n_random = (rewards_b == rewards_a).sum().item()
    choices[rewards_b == rewards_a] = np.random.randint(0, 2, (n_random,))
    return choices

np.random.seed(0)
N = 20000

options = np.random.rand(N, 2)
df = pd.DataFrame(options, columns=['option_a', 'option_b'])
for z in [0, 1]:
    df[f'reward_a_{z}'] = df.apply(lambda x: get_reward(x['option_a'], z), axis=1).astype(np.float32)
    df[f'reward_b_{z}'] = df.apply(lambda x: get_reward(x['option_b'], z), axis=1).astype(np.float32)
    df[f'choice_{z}'] = get_choices(df[f'reward_a_{z}'], df[f'reward_b_{z}'])


df['idx'] = np.arange(N)


dataset = Dataset.from_pandas(df[['idx', 'option_a', 'option_b']])
dataset.save_to_disk('data/synthetic_data/raw_pairs')


df['split'] = np.random.choice(['train', 'val', 'test'], N, p=[0.8, 0.1, 0.1])
df[['idx', 'split', 'reward_a_0', 'reward_b_0', 'choice_0', 'choice_1']].to_csv('data/synthetic_data/synthetic.csv', index=False)