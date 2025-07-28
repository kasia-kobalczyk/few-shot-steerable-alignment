import os
import sys
sys.path.append('../')
from pathlib import Path
from data.utils import setup_dataloaders, collect_pairs_choices
import torch
import matplotlib.pyplot as plt
import numpy as np
import scienceplots
import seaborn as sns
from tqdm import tqdm
from src.loss import DPOLoss
import pandas as pd
from src.models import LLMPolicy
from omegaconf import OmegaConf

def load_model_from_save(save_dir: str, model_class, device: torch.device):
    """Instantiate *model_class* from ``save_dir`` and load its *best* checkpoint."""
    cfg_path = Path(save_dir) / "config.yaml"
    ckpt_path = Path(save_dir) / "model_best.pt"

    if not cfg_path.exists():
        raise FileNotFoundError(cfg_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(ckpt_path)

    cfg = OmegaConf.load(str(cfg_path))
    model_cfg = cfg.model

    model = model_class(model_cfg)
    state_dict = torch.load(str(ckpt_path), map_location=device)
    
    # Print state dict keys that will be loaded
    print("\nState dict keys to load:")
    for key in state_dict.keys():
        print(f"  {key}")
        
    # Load state dict and get results
    load_result = model.load_state_dict(state_dict, strict=False)
    
    model.to(device)
    model.eval()
    return cfg, model



device='cuda:0'

root = '/mnt/pdata/caf83/few-shot-alignment/saves/dpo-ultrafeedback'

# Load the models
save_dict = {
    #'hh-btl-mixed' : f'{root}/hh-dpo-btl-mixed_1',
    #'hh-btl-honesty': f'{root}/hh-dpo-btl-honesty_0',
    #'hh-btl-helpfulness': f'{root}/hh-dpo-btl-helpfulness_0',
    'hh-nppl-mixed-new_1' : f'{root}/hh-dpo-nppl-mixed-new_0',
    # 'hht-btl-helpfulness': f'{root}/hht-btl-helpfulness_0',
    # 'hht-btl-honesty': f'{root}/hht-btl-honesty_0',
    # 'hht-btl-truthfulness': f'{root}/hht-btl-truthfulness_0',
    # 'hht-btl-mixed_0': f'{root}/hht-btl-mixed_0',
    # 'hht-nppl-mixed_0': f'{root}/hht-nppl-mixed_0',
}

num_context_ls = [0, 1, 3, 5, 10]

eval_dict = {}

for model_name, _ in save_dict.items():
    eval_dict[model_name] = {}
    for metric in ['accuracy', 'unseen_accuracy', 'label']:
        eval_dict[model_name][metric] = {}
        for num_context in num_context_ls:
            eval_dict[model_name][metric][num_context] = []

for model_name, model_path in save_dict.items():
    cfg, model = load_model_from_save(model_path, LLMPolicy, device)
    
    cfg.data.batch_size = 1
    cfg.data.max_num_context = 10
    cfg.data.num_targets= 13
    test_data_loader = setup_dataloaders(cfg.data, splits=['val'])['val']
    
    loss_func = DPOLoss(cfg.loss)
    
    
    print('Evaluting model:', model_name)
    for j, batch in enumerate(tqdm(test_data_loader)):
        for num_context in num_context_ls:
            
            pairs_C, choices_C, pairs_T, choices_T = collect_pairs_choices(
                batch, 
                num_context=num_context,
                min_num_context=cfg.data.min_num_context,
                max_num_context=cfg.data.max_num_context,
                num_targets=cfg.data.num_targets,
                context_datatype=cfg.data.context_datatype
            )  

            pairs_T = {k: v.to(device) for k,v in pairs_T.items()}
            choices_T = choices_T.to(device)
            pairs_C = pairs_C.to(device)
            choices_C = choices_C.to(device)

            with torch.no_grad():
                with torch.autocast(device_type="cuda"):
                    outputs = model(pairs_T, choices_T, pairs_C, choices_C)
                    
                    loss_values = loss_func(
                        logprobs=outputs['logprobs'], 
                        ref_logprobs=outputs['ref_logprobs'], 
                        choices=choices_T, 
                    )
            
            predictions = loss_values["accuracy_implicit_full"].unsqueeze(-1)
            choices = choices_T.unsqueeze(0).expand(predictions.shape[0], -1, -1, -1)
            acc = (predictions).float().mean(axis=0)

            bs = cfg.data.batch_size
            num_targets = cfg.data.num_targets

            unseen_predictions = torch.zeros(
                (predictions.shape[0], bs, num_targets - num_context, 1)
            )

            for i in range(bs):
                idx = torch.tensor(list(range(num_context, num_targets)))
                unseen_predictions[:, i, :, :] = predictions[:, i, idx, :]

            unseen_acc = (unseen_predictions).float().mean(axis=0)
        
            eval_dict[model_name]['accuracy'][num_context].append(acc)
            eval_dict[model_name]['unseen_accuracy'][num_context].append(unseen_acc)
            eval_dict[model_name]['label'][num_context].append(batch['labels_T'])
            
        if j > 500:
            break
            
        
num_context = 0

def get_res_df(eval_dict, acc_type="accuracy"):
    acc_dict = eval_dict[acc_type]
    labels_dict = eval_dict['label']
    res_df = pd.DataFrame()
    for num_context in num_context_ls:
        mean_acc = torch.stack(acc_dict[num_context]).squeeze(-1).mean(dim=-1).cpu().numpy()
        labels = [z for zs in labels_dict[num_context] for z in zs]
        try:
            res = pd.DataFrame({
                'acc' : mean_acc.flatten() * 100,
                'labels' : labels,
                'num_context' : num_context
            })
        except(ValueError):
            res = pd.DataFrame()
            print(acc_type, num_context)
        res_df = pd.concat([res_df, res])

    return res_df

res_df = pd.DataFrame()

for model_name in save_dict.keys():
    res_df_model = get_res_df(eval_dict[model_name], acc_type="unseen_accuracy" if 'nppl' in model_name else "accuracy")
    res_df_model['model'] = model_name
    res_df = pd.concat([res_df, res_df_model])
    
res_df.to_csv('../figures/dpo-all-models.csv', index=False)

res_df = pd.read_csv('../figures/dpo-all-models.csv')
summary_df = res_df.groupby(['model', 'num_context', 'labels'])['acc'].agg(['mean', 'sem']) 
summary_df['acc'] = summary_df.apply(lambda x: f"{x['mean']:.1f} ± {x['sem']:.1f}", axis=1)
final_df = summary_df.reset_index().pivot(index='num_context', columns=['model', 'labels'], values='acc')
print(final_df)

final_df.to_csv('../figures/dpo_summary.csv', index=False)