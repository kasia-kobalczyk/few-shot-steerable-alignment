import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import wandb
import numpy as np
from tqdm import tqdm
from argparse import ArgumentParser
import sys
import os

root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
sys.path.append(root)

from data.utils import setup_dataloaders, collect_pairs_choices
from models import RewardModel, LLMPolicy
from loss import RewardLoss, DPOLoss

EVAL_ITER = 500
SAVE_ITER = 500
MAX_EVAL_IT = 50

class Trainer:
    def __init__(self, cfg, save_dir):
        self.cfg = cfg
        self.device = cfg.model.device
        dataloaders = setup_dataloaders(cfg.data, splits=['train', 'val'])
        self.train_dataloader = dataloaders['train']
        self.val_dataloader = dataloaders['val']
        self.last_save_it = 0
        self.num_epochs = cfg.training.num_epochs
        if self.cfg.model.model_type == 'reward':
            self.model = RewardModel(cfg.model)
            self.loss_func = RewardLoss(cfg.loss)
        elif self.cfg.model.model_type == 'policy':
            self.model = LLMPolicy(cfg.model)
            self.loss_func = DPOLoss(cfg.loss)
        self.model.to(self.device)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=cfg.training.lr, eps=3e-5
        )

        self.params_to_save = []

        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.params_to_save.append(name)

    
        own_trainable_states = []
        print("Trainable parameters:")
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                print(name)
                own_trainable_states.append(name)

        self.save_dir = save_dir

    def run_batch(self, batch, num_context=None):

        pairs_C, choices_C, pairs_T, choices_T = collect_pairs_choices(
            batch, 
            num_context=num_context, 
            min_num_context=self.cfg.data.min_num_context, 
            max_num_context=self.cfg.data.max_num_context, 
            num_targets=self.cfg.data.num_targets, 
            context_datatype=self.cfg.data.context_datatype
        )
        
        if self.cfg.data.context_datatype == 'tokens':
            for k, v in pairs_C.items():
                pairs_C[k] = v.to(self.device)
        else:
            pairs_C = pairs_C.to(self.device)
        
        if self.cfg.data.target_datatype == 'tokens':
            for k, v in pairs_T.items():
                pairs_T[k] = v.to(self.device)
        else:
            pairs_T = pairs_T.to(self.device)

        choices_C = choices_C.to(self.device)
        choices_T = choices_T.to(self.device)

        with torch.autocast(device_type="cuda"):
            outputs = self.model(pairs_T, choices_T, pairs_C, choices_C)
            if self.cfg.model.model_type == 'reward':
                loss_values = self.loss_func(
                    logp_choices=outputs['logp_choices'],
                    choices=choices_T, 
                    q_z_C=outputs['q_z_C'], 
                    q_z_T=outputs['q_z_T']
                )
                predictions = outputs['logp_choices'].argmax(dim=-1, keepdim=True)
        
            else:
                loss_values = self.loss_func(
                    logprobs=outputs['logprobs'], 
                    ref_logprobs=outputs['ref_logprobs'], 
                    choices=choices_T, 
                )
                predictions = outputs['logprobs'].argmax(dim=-1, keepdim=True)
        
        choices_T = choices_T.unsqueeze(0).expand(predictions.shape[0], -1, -1, -1)
        accuracy = (predictions == choices_T).float().mean()
                

        bs = predictions.shape[1]
        num_targets = predictions.shape[2]
        if num_context is None:
            num_context = choices_C.shape[1]

        if num_targets == num_context:
            unseen_acc = torch.zeros(1)

        else:
            unseen_predictions = torch.zeros(
                (predictions.shape[0], bs, num_targets - num_context, 1)
            )
            unseen_choices = torch.zeros(
                (predictions.shape[0], bs, num_targets - num_context, 1)
            )

            for i in range(bs):
                idx = torch.tensor(list(range(num_context, num_targets)))
                unseen_predictions[:, i, :, :] = predictions[:, i, idx, :]
                unseen_choices[:, i, :, :] = choices_T[:, i, idx, :]

            unseen_acc = (unseen_predictions == unseen_choices).float().mean()

        results = loss_values
        results["accuracy"] = accuracy
        results["unseen_acc"] = unseen_acc

        return results

    def train(self):
        it = 0
        min_eval_loss = np.inf
        for epoch in range(self.num_epochs + 1):
            print(f"Epoch {epoch}/{self.num_epochs}")
            for batch in tqdm(self.train_dataloader):
                self.model.train()
                self.optimizer.zero_grad()
                results = self.run_batch(batch)
                loss = results["loss"]
                loss.backward()
                self.optimizer.step()
                if not self.cfg.training.dry_run:
                    wandb.log(dict(zip(
                        ['train_' + k for k in results.keys()],
                        results.values()
                    )))

                if it % (EVAL_ITER // self.cfg.data.batch_size) == 0 and it > 0:
                    eval_dict, val_loss = self.eval()
                    if not self.cfg.training.dry_run:
                        wandb.log({
                            "eval_loss": val_loss
                        })
                        for metric in eval_dict.keys():
                            for num_context, value in eval_dict[metric].items():
                                wandb.log({
                                    f"eval_{metric}_{num_context}": value
                                })
                        
                        if val_loss < min_eval_loss:
                            min_eval_loss = val_loss

                            state_dict_save = self.model.state_dict()

                            torch.save(state_dict_save, f"{self.save_dir}/model_best.pt")
                            torch.save(
                                self.optimizer.state_dict(),
                                f"{self.save_dir}/optim_best.pt",
                            )
                            print(f"Best model saved at iteration {self.last_save_it + it}")
                it += 1

        return min_eval_loss

    def eval(self):
        print("Evaluating")
        it = 0
        self.model.eval()
        with torch.no_grad():
            loss_num_context = [0, 1, 3, 5]
            if self.cfg.data.min_num_context == 0:
                loss_num_context = [0] + loss_num_context
            
            eval_metrics = ['loss', 'accuracy', 'unseen_acc']
            if self.cfg.model.model_type == 'policy':
                eval_metrics += ['accuracy_implicit']

            eval_dict = dict(zip(eval_metrics, loss_num_context))
            for metric in eval_metrics:
                eval_dict[metric] = {}
                for num_context in loss_num_context: 
                    eval_dict[metric][num_context] = []

            val_losses = []

            for batch in self.val_dataloader:
                for num_context in loss_num_context:
                    results = self.run_batch(batch, num_context=num_context)
                    
                    for metric in eval_metrics:
                        eval_dict[metric][num_context].append(results[metric].to("cpu").item())
                    
                    val_results = self.run_batch(batch)
                    val_loss = val_results["loss"]
                    val_losses.append(val_loss.to("cpu").item())

                it += 1
                if it > (MAX_EVAL_IT // self.cfg.data.batch_size):
                    break
                    
            for metric in eval_metrics:
                for num_context in loss_num_context:
                    eval_dict[metric][num_context] = np.mean(eval_dict[metric][num_context])
           
            val_loss = np.mean(val_losses)

        return eval_dict, val_loss


@hydra.main(version_base=None, config_path=f"{root}/config", config_name="config")
def train(cfg):
    
    np.random.seed(cfg.training.seed)
    torch.manual_seed(cfg.training.seed)
    torch.cuda.manual_seed(cfg.training.seed)
    
    if not cfg.training.dry_run:
        # Create save folder and save cfg
        run_name_prefix = cfg.save.run_name_prefix if cfg.save.run_name_prefix else "run"
        save_dir = f"saves/{cfg.save.project_name}"
        os.makedirs(save_dir, exist_ok=True)
        save_no = len(os.listdir(save_dir))
        save_no = [
            int(x.split("_")[-1])
            for x in os.listdir(save_dir)
            if x.startswith(run_name_prefix)
        ]
        if len(save_no) > 0:
            save_no = max(save_no) + 1
        else:
            save_no = 0
        save_dir = os.path.join(save_dir, f"{run_name_prefix}_{save_no}")
        os.makedirs(save_dir, exist_ok=True)
        
        trainer = Trainer(cfg=cfg, save_dir=save_dir)
        # Save cfg
        cfg = trainer.cfg
        with open(f"{save_dir}/config.yaml", "w") as f:
            OmegaConf.save(cfg, f)
            
        # Initialize wandb
        wandb.config = OmegaConf.to_container(
            cfg, resolve=True, throw_on_missing=True
        )
        wandb.init(
            project=cfg.save.project_name, name=f"{run_name_prefix}_{save_no}"
        )
        best_eval_loss = trainer.train()
        wandb.finish()
    
    else:
        trainer = Trainer(cfg=cfg, save_dir=None)
        best_eval_loss = trainer.train()

    return best_eval_loss



if __name__ == "__main__":
    train()
    

    