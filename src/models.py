import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
from torch.nn.modules.transformer import MultiheadAttention
from trl.models.modeling_base import create_reference_model
from trl.trainer.utils import selective_log_softmax
import os 
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.modules import MLP, ConditionalLLM
from src.utils import MultivariateNormalDiag, setup_llm, print_trainable_parameters


class RewardModel(nn.Module):
    def __init__(self, cfg):
        super(RewardModel, self).__init__()
        self.cfg = cfg
        if cfg.is_conditional:
            self.model = ConditionalRewardModel(cfg)
        else:
            self.model = SimpleRewardModel(cfg)

    def forward(self, pairs_T, choices_T, pairs_C=None, choices_C=None):
        if self.cfg.is_conditional:
            rewards, z_C, q_z_C, q_z_T = self.model(pairs_T, choices_T, pairs_C, choices_C)
        else:
            rewards = self.model(pairs_T, choices_T)
            z_C = None
            q_z_C = None
            q_z_T = None
            z_C = None
        
        logp_choices = self.get_logprobs(rewards) # shape (num_z_samples, batch_size, num_targets, 1)
        
        outputs = {
            'rewards': rewards,
            'logp_choices': logp_choices,
            'q_z_C': q_z_C,
            'q_z_T': q_z_T,
            'z_C': z_C
        }
        return outputs

    def get_logprobs(self, rewards):
        if self.cfg.distributional_head:
            rewards_mean, rewards_std = rewards.split(1, dim=-1)
            rewards_mean_a, rewards_mean_b = rewards_mean.squeeze(-1).split(1, dim=-1)
            rewards_std_a, rewards_std_b = rewards_std.squeeze(-1).split(1, dim=-1)
            diff_means = rewards_mean_a - rewards_mean_b
            vars_combined = rewards_std_a**2 + rewards_std_b**2
            z = diff_means / torch.sqrt(vars_combined)
            logp_choose_a = -F.softplus(-z * np.sqrt(2 * np.pi))
            logp_choose_b = -F.softplus(z * np.sqrt(2 * np.pi))
            logp_choices = torch.cat([logp_choose_a, logp_choose_b], dim=-1)

        else:
            rewards_a, rewards_b = rewards.squeeze(-1).split(1, dim=-1)
            logp_choose_a = F.logsigmoid(rewards_a - rewards_b)
            logp_choose_b = F.logsigmoid(rewards_b - rewards_a)
            logp_choices = torch.cat([logp_choose_a, logp_choose_b], dim=-1)

        return logp_choices


class SimpleRewardModel(nn.Module):
    def __init__(self, cfg):
        super(SimpleRewardModel, self).__init__()
        self.cfg = cfg
        self.target_encoder = TargetEncoder(cfg)
        self.decoder = MLP(
            input_size=cfg.hidden_dim,
            hidden_size=cfg.hidden_dim,
            num_hidden=2,
            output_size=1 if not cfg.distributional_head else 2,
            activation=nn.ReLU(),
            batch_norm=True
        )
    
    def forward(self, pairs_T, choices_T):
        """
        pairs_T: tensor of shape (batch_size, num_targets, 2, input_dim)
        choices_T: tensor of shape (batch_size, num_targets, 1)
        """
        encoded_pairs_T = self.target_encoder(pairs_T) # shape (batch_size, num_targets, 2, hidden_dim)
        bs, num_targets, _ = choices_T.size()
        rewards = self.decoder(encoded_pairs_T.view(-1, self.cfg.hidden_dim)) # shape (batch_size * num_targets * 2, 1)
        rewards = rewards.view(bs, num_targets, 2, -1).unsqueeze(0) # shape (1, batch_size, num_targets, 2, 1)

        if self.cfg.distributional_head:
            mean_and_std = rewards
            mean, std = mean_and_std.split(1, dim=-1)
            std = F.softplus(std)
            rewards = torch.cat([mean, std], dim=-1)

        return rewards


class ConditionalRewardModel(nn.Module):
    def __init__(self, cfg):
        super(ConditionalRewardModel, self).__init__()
        self.cfg = cfg
        self.context_encoder = ContextEncoder(cfg)
        self.target_encoder = TargetEncoder(cfg)
        self.decoder = RewardDecoder(cfg)
        
    def forward(self, pairs_T, choices_T, pairs_C, choices_C):
        """
        pairs_T: tensor of shape (batch_size, num_targets, 2, input_dim)
        choices_T: tensor of shape (batch_size, num_targets, 1)
        context: tensor of shape (batch_size, num_context, 2, input_dim)
        choices_C: tensor of shape (batch_size, num_context, 1)
        """

        encoded_pairs_T = self.target_encoder(pairs_T) # shape (batch_size, num_targets, 2, hidden_dim)

        z_C, q_z_C, q_z_T = self.context_encoder(pairs_C, choices_C, pairs_T, choices_T) # shape (num_z_samples, batch_size, hidden_dim)

        rewards = self.decoder(encoded_pairs_T, z_C) # shape (num_z_samples, batch_size, num_targets, 2, output_dim)

        return rewards, z_C, q_z_C, q_z_T


class LLMPolicy(nn.Module):
    def __init__(self, cfg):
        super(LLMPolicy, self).__init__()
        self.cfg = cfg
        if cfg.is_conditional:
            self.model = ConditionalLLMPolicy(cfg)
        else:
            self.model = SimpleLLMPolicy(cfg)
    
    def forward(self, pairs_T, choices_T, pairs_C=None, choices_C=None):
        if self.cfg.is_conditional:
            outputs = self.model(pairs_T, choices_T, pairs_C, choices_C)
        else:
            outputs = self.model(pairs_T, choices_T)

        logprobs = self.get_logprobs(outputs['logprobs'])
        ref_logprobs = self.get_logprobs(outputs['ref_logprobs'])
        outputs = {
            'logprobs': logprobs,
            'ref_logprobs': ref_logprobs
        }
        return outputs
    
    def get_logprobs(self, logprobs):
        logprobs_a, logprobs_b = logprobs.split(1, dim=-2) # (num_z_samples, batch_size, num_targets, max_len)
        logprobs_a, logprobs_b = logprobs_a.squeeze(-2), logprobs_b.squeeze(-2)
        logprobs = torch.cat([logprobs_a, logprobs_b], dim=-1) # (num_z_samples, batch_size, num_targets, 2)
        return logprobs
    
    def generate(self, *args, **kwargs):
        if self.cfg.is_conditional:
            return self.model.generate(*args, **kwargs)
        else:
            return self.model.model.generate(*args, **kwargs)


class SimpleLLMPolicy(nn.Module):
    def __init__(self, cfg):
        super(SimpleLLMPolicy, self).__init__()
        self.cfg = cfg
        self.model = setup_llm(
            cfg.llm_name,
            causal=True,
            is_lora_tunable=cfg.tune_llm,
            lora_kwargs={
                'alpha': cfg.lora_alpha,
                'dropout': cfg.lora_dropout,
                'r': cfg.lora_r
            }
        )
        self.reference_model = create_reference_model(self.model)

    def get_logprobs(self, model, input_ids, attention_mask, response_start_idx, response_end_idx):
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        logits = outputs.logits
        
        loss_mask = torch.zeros_like(input_ids, dtype=torch.bool)
        for i in range(input_ids.shape[0]):
            loss_mask[i, response_start_idx[i]:response_end_idx[i]] = True
        
         # Offset the logits by one to align with the labels
        labels = torch.roll(input_ids, shifts=-1, dims=1)
        loss_mask = torch.roll(loss_mask, shifts=-1, dims=1).bool()
        
        labels[~loss_mask] = 0  # dummy token; we'll ignore the losses on these tokens later
        per_token_logps = selective_log_softmax(logits, labels)
        per_token_logps[~loss_mask] = 0
        per_token_logps = torch.roll(per_token_logps, shifts=1, dims=1)
        return per_token_logps[:, 1:].sum(-1)

    def forward(self, pairs_T, choices_T):
        input_ids = pairs_T['input_ids'] # (batch_size, num_targets, 2, max_len)
        attention_mask = pairs_T['attention_mask'] # (batch_size, num_targets, 2, max_len)
        input_ids = input_ids.view(-1, input_ids.shape[-1])
        attention_mask = attention_mask.view(-1, attention_mask.shape[-1])
        response_start_idx = pairs_T['response_start_idx'].view(-1) # (batch_size * num_targets)
        response_end_idx = pairs_T['cls_idx'].view(-1) # (batch_size * num_targets)

        bs, num_targets, _, _ = pairs_T['input_ids'].shape

        with torch.no_grad():
            ref_logprobs = self.get_logprobs(
                model=self.reference_model,
                input_ids=input_ids,
                attention_mask=attention_mask,
                response_start_idx=response_start_idx,
                response_end_idx=response_end_idx
            )
            ref_logprobs = ref_logprobs.view(bs, num_targets, 2, -1).unsqueeze(0)
        
        logprobs = self.get_logprobs(
            model=self.model,
            input_ids=input_ids,
            attention_mask=attention_mask,
            response_start_idx=response_start_idx,
            response_end_idx=response_end_idx
        )
        logprobs = logprobs.view(bs, num_targets, 2, -1).unsqueeze(0)
        outputs = {
            'logprobs': logprobs,
            'ref_logprobs': ref_logprobs
        }
        return outputs


class ConditionalLLMPolicy(nn.Module):
    def __init__(self, cfg):
        super(ConditionalLLMPolicy, self).__init__()
        self.cfg = cfg
        self.context_encoder = ContextEncoder(cfg)
        self.decoder = ConditionalPolicyDecoder(cfg)
        
    def forward(self, pairs_T, choices_T, pairs_C, choices_C):
        z_C, _, _ = self.context_encoder(pairs_C, choices_C, pairs_T, choices_T)
        raw_logprobs = self.decoder(pairs_T, z_C)
        raw_ref_logprobs = self.decoder(pairs_T, z_C, ref_model=True)
        outputs = {
            'logprobs': raw_logprobs,
            'ref_logprobs': raw_ref_logprobs
        }
        return outputs

    def get_latent_var(self, pairs_T, choices_T, pairs_C, choices_C):
        z_C, _, _ = self.context_encoder(pairs_C, choices_C, pairs_T, choices_T)
        return z_C
    
    def generate(self, *args, **kwargs):
        z_C, _, _ = self.context_encoder(*args[:4])
        return self.decoder.conditional_llm.conditional_generate(z_C.flatten(1), args[4], **kwargs)
        
        


class ConditionalPolicyDecoder(nn.Module):
    def __init__(self, cfg):
        super(ConditionalPolicyDecoder, self).__init__()
        self.conditional_llm = ConditionalLLM(
            llm_name=cfg.llm_name,
            num_film_layers=cfg.num_film_layers,
            is_lora_tunable=cfg.tune_llm,
            lora_kwargs={
                'alpha': cfg.lora_alpha,
                'dropout': cfg.lora_dropout,
                'r': cfg.lora_r
            },
            hidden_dim=cfg.hidden_dim
        )
        self.reference_model = create_reference_model(self.conditional_llm.base_model)

    def forward(self, pairs_T, z_C, ref_model=False):
        """
        pairs_T: tensor of shape (batch_size, num_targets, 2, max_len)
        z_C: tensor of shape (num_z_samples, batch_size, hidden_dim)
        """
        num_z_samples, bs, hidden_dim = z_C.size()
        _, num_targets, _, max_len = pairs_T['input_ids'].shape
        z_C = z_C.unsqueeze(1).unsqueeze(2).view(num_z_samples, bs, 1, 1, hidden_dim).expand(-1, -1, num_targets, 2, -1)
        assert z_C.size() == (num_z_samples, bs, num_targets, 2, hidden_dim)
        z_C = z_C.reshape(-1, hidden_dim)
        input_ids = pairs_T['input_ids'] # (batch_size, num_targets, 2, max_len)
        attention_mask = pairs_T['attention_mask'] # (batch_size, num_targets, 2, max_len)
        input_ids = input_ids.view(-1, input_ids.shape[-1])
        attention_mask = attention_mask.view(-1, attention_mask.shape[-1])
        response_start_idx = pairs_T['response_start_idx'].view(-1) # (batch_size * num_targets)
        response_end_idx = pairs_T['cls_idx'].view(-1) # (batch_size * num_targets)

        
        if ref_model:
            with torch.no_grad():
                outputs = self.reference_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
        else:
            outputs = self.conditional_llm(
                input_ids=input_ids,
                attention_mask=attention_mask, 
                latent_variable=z_C
            )
        
        logits = outputs.logits
        
        loss_mask = torch.zeros_like(input_ids, dtype=torch.bool)
        for i in range(input_ids.shape[0]):
            loss_mask[i, response_start_idx[i]:response_end_idx[i]] = True
        
         # Offset the logits by one to align with the labels
        labels = torch.roll(input_ids, shifts=-1, dims=1)
        loss_mask = torch.roll(loss_mask, shifts=-1, dims=1).bool()
        
        labels[~loss_mask] = 0  # dummy token; we'll ignore the losses on these tokens later
        per_token_logps = selective_log_softmax(logits, labels)
        per_token_logps[~loss_mask] = 0
        per_token_logps = torch.roll(per_token_logps, shifts=1, dims=1)

        return per_token_logps[:, 1:].sum(-1).view(num_z_samples, bs, num_targets, 2, -1)


class TargetEncoder(nn.Module):
    def __init__(self, cfg):
        super(TargetEncoder, self).__init__()
        self.cfg = cfg
        if cfg.target_input_encoder == 'mlp':
            self.inner_encoder = MLP(
                input_size=cfg.input_dim,
                hidden_size=cfg.hidden_dim,
                num_hidden=2,
                output_size=cfg.hidden_dim,
                activation=nn.ReLU(),
                batch_norm=False
            )
        elif cfg.target_input_encoder == 'linear':
            self.inner_encoder = nn.Linear(cfg.input_dim * 2, cfg.hidden_dim)
        else:
            self.inner_encoder = setup_llm(
                cfg.target_input_encoder,
                causal=False,
                is_lora_tunable=cfg.tune_llm,
                lora_kwargs={
                    'alpha': cfg.lora_alpha,
                    'dropout': cfg.lora_dropout,
                    'r': cfg.lora_r
                }
            )
            self.outer_encoder = nn.Linear(self.inner_encoder.config.hidden_size, cfg.hidden_dim)
    
    def forward(self, pairs_T):
        """
        pairs_T: tensor of shape (batch_size, num_targets, 2, input_dim)
        """
        if self.cfg.target_input_encoder in ['mlp', 'linear']:
            bs, num_targets, _, _ = pairs_T.shape
            x = pairs_T.view(-1, pairs_T.shape[-1])
            x = self.inner_encoder(x)
            x = x.view(bs, num_targets, 2, -1)
        else:
            bs, num_targets, _, _ = pairs_T['input_ids'].shape
            input_ids = pairs_T['input_ids'] # (batch_size, num_targets, 2, max_len)
            attention_mask = pairs_T['attention_mask'] # (batch_size, num_targets, 2, max_len)
            cls_idxs = pairs_T['cls_idx'] # (batch_size, num_targets, 2)
            input_ids = input_ids.view(-1, input_ids.shape[-1])
            attention_mask = attention_mask.view(-1, attention_mask.shape[-1])
            cls_idxs = cls_idxs.view(-1)

            last_hidden_state = self.inner_encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True
            ).hidden_states[-1]
            x = torch.zeros(input_ids.shape[0], last_hidden_state.shape[-1], device=input_ids.device)
            for i in range(input_ids.shape[0]):
                x[i, :] = last_hidden_state[i, cls_idxs[i], :]
            x = self.outer_encoder(x)
            x = x.view(bs, num_targets, 2, -1)
        return x


class ContextEncoder(nn.Module):
    def __init__(self, cfg):
        super(ContextEncoder, self).__init__()
        self.cfg = cfg
        if cfg.context_input_encoder == 'mlp':
            self.inner_encoder = MLP(
                input_size=cfg.input_dim * 2,
                hidden_size=cfg.hidden_dim,
                num_hidden=2,
                output_size=cfg.hidden_dim,
                activation=nn.ReLU(),
                batch_norm=False
            )
        elif cfg.context_input_encoder == 'linear':
            self.inner_encoder = nn.Linear(cfg.input_dim * 2, cfg.hidden_dim)

        if cfg.context_agg_func == 'self-attention':
            self.self_attention = MultiheadAttention(
                cfg.hidden_dim, 8, 0.1, batch_first=True
            )

        self.outer_encoder = MLP(
            input_size=cfg.hidden_dim,
            hidden_size=cfg.hidden_dim,
            num_hidden=2,
            output_size=cfg.hidden_dim if not cfg.sample_latents else 2 * cfg.hidden_dim,
            activation=nn.ReLU(),
            batch_norm=False
        )

    def encode_globally(self, pairs, choices):
        """
        pairs: tensor of shape (batch_size, num_pairs, 2, input_dim)
        choices: tensor of shape (batch_size, num_pairs, 1)
        """
        bs, num_pairs, _, _ = pairs.size()

        chosen = torch.where(choices == 0, pairs[:, :, 0, :], pairs[:, :, 1, :])
        rejected = torch.where(choices == 1, pairs[:, :, 0, :], pairs[:, :, 1, :])
        x = torch.cat([chosen, rejected], dim=-1)
        x = self.inner_encoder(x) # shape (batch_size, num_pairs, hidden_dim)
        if self.cfg.context_agg_func == 'self-attention':
            x = self.self_attention(x, x, x)[0]
            x = x.sum(dim=1)
        elif self.cfg.context_agg_func == 'mean':
            x = x.mean(dim=1)
        elif self.cfg.context_agg_func == 'sum':
            x = x.sum(dim=1)
        x = self.outer_encoder(x) # shape (batch_size, hidden_dim)
        return x
    
    def forward(self, pairs_C, choices_C, pairs_T, choices_T):
        """
        pairs_C: tensor of shape (batch_size, num_context, 2, input_dim)
        choices_C: tensor of shape (batch_size, num_context, 1)
        pairs_T: tensor of shape (batch_size, num_targets, 2, hidden_dim)
        choices_T: tensor of shape (batch_size, num_targets, 1)
        """
        if self.cfg.sample_latents:
            latent_params_C = self.encode_globally(pairs_C, choices_C)
            latent_params_T = self.encode_globally(pairs_T, choices_T)
            mean_C, logvar_C = latent_params_C.split(self.cfg.hidden_dim, dim=-1)
            mean_T, logvar_T = latent_params_T.split(self.cfg.hidden_dim, dim=-1)
            q_z_C = MultivariateNormalDiag(mean_C, F.softplus(logvar_C))
            q_z_T = MultivariateNormalDiag(mean_T, F.softplus(logvar_T))
            if self.training:
                z_C = q_z_C.rsample([self.cfg.num_z_samples_train])
            else:
                z_C = q_z_C.sample([self.cfg.num_z_samples_eval])

        else:
            z_C = self.encode_globally(pairs_C, choices_C).unsqueeze(0)
            q_z_C = None
            q_z_T = None

        return z_C, q_z_C, q_z_T


class RewardDecoder(nn.Module):
    def __init__(self, cfg):
        super(RewardDecoder, self).__init__()
        self.cfg = cfg
        if cfg.distributional_head:
            self.output_dim = 2 * cfg.input_dim
        else:
            self.output_dim = 1

        self.decoder = MLP(
            input_size=cfg.hidden_dim * 2,
            hidden_size=cfg.hidden_dim,
            num_hidden=2,
            output_size=self.output_dim,
            activation=nn.ReLU(),
            batch_norm=True
        )

    def forward(self, pairs_T, z_C):
        """
        pairs_T: tensor of shape (batch_size, num_targets, 2, hidden_dim)
        z_C: tensor of shape (num_z_samples, batch_size, hidden_dim)
        """
        bs, num_targets, _, _ = pairs_T.size()
        pairs_T = pairs_T.view(bs, num_targets * 2, -1)
        pairs_T = pairs_T.unsqueeze(0).expand(z_C.size(0), -1, -1, -1)
        z_C = z_C.unsqueeze(2).expand(-1, -1, num_targets * 2, -1)
        x = torch.cat([pairs_T, z_C], dim=-1)
        x = self.decoder(x.view(-1, x.shape[-1]))
        x = x.view(z_C.size(0), bs, num_targets, 2, -1)

        if self.cfg.distributional_head:
            mean_and_std = x
            mean, std = mean_and_std.split(1, dim=-1)
            std = F.softplus(std)
            x = torch.cat([mean, std], dim=-1)

        return x



if __name__ == '__main__':
    from hydra import compose, initialize
    from omegaconf import OmegaConf
    import os
    from data.utils import setup_dataloaders


    with initialize(config_path="./config"):
        cfg = compose(config_name="hh_dpo_config")
    

    train_dataloader = setup_dataloaders(cfg.data, splits=['train'])['train']
    device = cfg.model.device
    model = ConditionalLLMPolicy(cfg.model).to(device)

    for batch in train_dataloader:
        pairs_C, choices_C, pairs_T, choices_T = batch['pairs_C'], batch['choices_C'], batch['pairs_T'], batch['choices_T']
        if cfg.data.context_datatype == 'tokens':
            for k, v in pairs_C.items():
                pairs_C[k] = v.to(device)
        else:
            pairs_C = pairs_C.to(device)
        
        if cfg.data.target_datatype == 'tokens':
            for k, v in pairs_T.items():
                pairs_T[k] = v.to(device)
        choices_C = choices_C.to(device)
        choices_T = choices_T.to(device)
        outputs = model(pairs_T, choices_T, pairs_C, choices_C)
        print(outputs['logprobs'].shape)
        break