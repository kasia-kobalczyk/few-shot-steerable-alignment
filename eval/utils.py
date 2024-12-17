import os
import sys
import torch
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.models import RewardModel, LLMPolicy
from hydra.experimental import compose, initialize
from omegaconf import OmegaConf

def _load_model(cfg, save_dir, load_it, device):
    print(save_dir)
    if cfg.model.model_type == 'reward':
        model = RewardModel(cfg.model)
    elif cfg.model.model_type == 'policy':
        model = LLMPolicy(cfg.model)
    model.to(device)
    model.eval()
    state_dict = torch.load(f'{save_dir}/model_{load_it}.pt', map_location=device)
    model.load_state_dict(state_dict)
    return model

def load_model(save_dir, load_it='best', device='cuda:0'):
    cfg = OmegaConf.load(f'{save_dir}/config.yaml')
    model = _load_model(cfg, save_dir, load_it=load_it, device=device)
    return model, cfg
