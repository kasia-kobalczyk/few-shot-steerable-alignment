
import torch.nn as nn
import torch
import torch.nn.functional as F

class RewardLoss(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.beta = cfg.beta_kl

    def forward(self, logp_choices, choices, q_z_C, q_z_T):
        """
        Compute the ELBO loss during training and NLL for validation and testing

        logp_choose_a: tensor of shape (num_z_samples, batch_size, num_targets, 2)
        choices: tensor of shape (batch_size, num_targets, 1)
        q_z_C: Normal distribution over z_C
        q_z_T: Normal distribution over z_T
        """
        # 1st term: negative log likelihood
        num_z_samples, bach_size, num_targets, _ = logp_choices.shape
        choices = choices.unsqueeze(0).expand(num_z_samples, -1, -1, -1)
        negative_ll = torch.functional.F.cross_entropy(
            logp_choices.view(-1, 2), choices.flatten().long(), reduction="none"
        )
        negative_ll = negative_ll.view(num_z_samples, bach_size, num_targets)
        negative_ll = negative_ll.mean(axis=0).mean(axis=-1)

        # 2nd term: KL[q(z | T) || q (z || C)]
        if q_z_T is not None:
            kl_z = torch.distributions.kl.kl_divergence(q_z_C, q_z_T)  # [batch_size]
            loss = negative_ll + self.beta * kl_z

        else:
            kl_z = None
            loss = negative_ll

        loss = loss.mean()
        negative_ll = negative_ll.mean()
        if kl_z is not None:
            kl_z = kl_z.mean()

        return {
            "loss": loss,
            "negative_ll": negative_ll,
            "kl_z": kl_z
        }


class DPOLoss(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.beta = cfg.beta_dpo
    
    def _get_logprob_ratios(self, logprobs, choices):
        logprobs_a, logprobs_b = logprobs.split(1, dim=-1) # (num_z_samples, batch_size, num_targets, 1)
        logprobs_a = logprobs_a.squeeze(-1)
        logprobs_b = logprobs_b.squeeze(-1)
        choices = choices.unsqueeze(0).expand(logprobs.shape[0], -1, -1, -1).squeeze(-1)
        chosen_logprobs = logprobs_a.clone()
        chosen_logprobs[choices == 1] = logprobs_b[choices == 1]
        rejected_logprobs = logprobs_b.clone()
        rejected_logprobs[choices == 1] = logprobs_a[choices == 1]
        return chosen_logprobs, rejected_logprobs

    def forward(self, logprobs, ref_logprobs, choices):
        """
        logprobs.shape (num_z_samples, batch_size, num_targets, 2)
        ref_logprobs.shape (num_z_samples, batch_size, num_targets, 2)
        choices.shape (batch_size, num_targets, 1)
        response_start_idx.shape (batch_size, num_targets, 2)
        cls_idx.shape (batch_size, num_targets)
        """
        pi_chosen_logprobs, pi_rejected_logprobs = self._get_logprob_ratios(logprobs, choices)
        ref_chosen_logprobs, ref_rejected_logprobs = self._get_logprob_ratios(ref_logprobs, choices)
        pi_logratios = pi_chosen_logprobs - pi_rejected_logprobs
        ref_logratios = ref_chosen_logprobs - ref_rejected_logprobs
        logits = pi_logratios - ref_logratios
        losses = -F.logsigmoid(self.beta * logits)

        implicit_rewards_chosen = self.beta * (pi_chosen_logprobs - ref_chosen_logprobs)
        implicit_rewards_rejected = self.beta * (pi_rejected_logprobs - ref_rejected_logprobs)     
        accuracy_implicit = (implicit_rewards_chosen > implicit_rewards_rejected).float().mean()

        return {
            "loss": losses.mean(),
            "pi_logratios": pi_logratios.mean(),
            "ref_logratios": ref_logratios.mean(),
            "accuracy_implicit" : accuracy_implicit,
            "accuracy_implicit_full" : implicit_rewards_chosen > implicit_rewards_rejected
        }

if __name__ == "__main__":
    from argparse import Namespace

    cfg = Namespace(
        beta_dpo=0.1
    )

    loss = DPOLoss(cfg)

    logprobs = torch.randn(1, 1, 10, 2)
    ref_logprobs = torch.randn(1, 1, 10, 2)
    choices = torch.randint(0, 2, (1, 10, 1))

    loss_value = loss(logprobs, ref_logprobs, choices)
    print(loss_value[0])