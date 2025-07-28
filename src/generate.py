#!/usr/bin/env python
"""Generate answers from a trained policy model and score them with two reward models.

Usage (examples)
----------------

# Simple (un‑conditional) policy model
python generate_policy_answers.py \
    --policy_save_dir /path/to/policy/run_3 \
    --help_reward_dir /path/to/reward_helpfulness/run_7 \
    --honesty_reward_dir /path/to/reward_honesty/run_2 \
    --split test \
    --out_dir ./generations/simple

# Conditional policy model (few‑shot conditioning)
python generate_policy_answers.py \
    --policy_save_dir /path/to/cond_policy/run_12 \
    --help_reward_dir /path/to/reward_helpfulness/run_7 \
    --honesty_reward_dir /path/to/reward_honesty/run_2 \
    --split test \
    --out_dir ./generations/conditional \
    --context_lengths 1 3 5 10

The script automatically detects whether the policy is conditional or not from the
stored config (cfg.model.is_conditional).  For conditional models it will loop over
all requested context lengths and generate a separate set of responses for each.

Outputs
-------
* ``<out_dir>/<split>_generations.jsonl`` – one JSON line per example (& per context length
   when conditional).  Each line contains::

       {
           "id":            <dataset row id>,
           "prompt":        <original prompt string>,
           "response":      <generated response string>,
           "context_len":   <int, 0 for simple models>,
           "help_score":    <reward model score>,
           "honesty_score": <reward model score>
       }

Notes
-----
* The script **only reads** from the saved training directories – no need to hand‑write
  any configs.  We assume each save directory contains a ``config.yaml`` and a
  ``model_best.pt`` (the convention used in ``Trainer``).
* Generation respects the chat template that was used for tokenisation during
  training (`apply_chat_template` if the base model supports it, otherwise the
  fallback template ``"Human: {prompt}\n\nAssistant:"``).
* Reward models are evaluated on the *pair* (prompt, generated response) exactly
  as during training: prompt & response are merged with the same chat template and
  fed through the reward network.  No gradients are computed.
* The script purposefully keeps the implementation minimal and self‑contained – it
  imports only *public* utilities from the existing code‑base (``data.utils``,
  ``models``) and never touches private trainer internals.
"""


import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Sequence
import numpy as np
import random

import torch
from omegaconf import OmegaConf
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel

root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
sys.path.append(root)


# Local modules (they live in the repo root)
from data.utils import setup_dataloaders  # noqa: E402
from models import RewardModel, LLMPolicy  # noqa: E402

################################################################################
# Utility helpers
################################################################################

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


def build_chat_prompt(tokenizer, prompt_text: str, system_prompt: str = "You are a helpful AI assistant.", add_generation_prompt=True) -> str:
    """Return a *text* (not tokenised) prompt using the correct chat template."""
    if hasattr(tokenizer, "apply_chat_template"):
        messages = [
            #{"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt_text},
        ]
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=add_generation_prompt)
    else:
        print("Using fallback chat template.d")
        bos = tokenizer.bos_token or ""
        return f"{bos}Human: {prompt_text}\n\nAssistant:"
        
        
def build_chat_prompt_reward(prompt_text: str, response_text: str) -> str:
    prompt_template = "Human: {prompt}\\n\\nAssistant: "
    template = prompt_template + "{response}"
    output = template.format(prompt=prompt_text, response=response_text)
    return output


def clean_prompt(text: str) -> str:
    """Extract user content from chat template."""
    # Find the content between 'user\n' and '\nassistant\n'
    if '\nuser\n' in text:
        content = text.split('\nuser\n')[1].split('\nassistant\n')[0]
    elif 'user\n' in text:
        content = text.split('user\n')[1].split('\nmodel\n')[0]
    else:
        # Fallback for other formats
        content = text.split('Human: ')[1].split('\n\nAssistant:')[0]
    return content.strip()


################################################################################
# Generation helpers
################################################################################

def generate_simple(policy: LLMPolicy, tokenizer, prompts, gen_kwargs: Dict) -> List[str]:
    """Generate *responses* for a batch of *prompts* using an **un‑conditional** policy."""
    device = policy.model.model.device  # type: ignore[attr-defined]
    # Tokenise *only* the user side of the conversation.
    
    model_inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding='max_length',
        truncation=True,
        max_length=512
    ).to(device)
    # ``LLMPolicy.generate`` simply forwards to the wrapped *transformers* model.
    with torch.autocast(torch.device(device).type):
        generated = policy.generate(**model_inputs, **gen_kwargs)

    decoded = tokenizer.batch_decode(generated, skip_special_tokens=True)
    responses: List[str] = []
    for prompt_text, text in zip(prompts, decoded):
        # Cut everything *before* the assistant turn so that we return only the response.
        anchor = "\nassistant\n"
        idx = text.rfind(anchor)
        if idx != -1:
            responses.append(text[idx + len(anchor):].strip())
        else:
            # Fallback – return everything after the original prompt string.
            after_prompt = text.split(prompt_text, 1)[-1]
            responses.append(after_prompt.strip())
    return responses


################################ CONDITIONAL ##################################

def _slice_tensor_dict(t: Dict[str, torch.Tensor], n: int):
    """Return a *view* of the first *n* context examples across the *batch* axis (dim=1).

    Works for the nested tensor structure produced by *tokenised* dataloaders.
    """
    out = {}
    for k, v in t.items():
        # Expected shape: (batch_size, num_context, 2, ...)
        out[k] = v[:, :n].contiguous()
    return out


def generate_conditional(
    policy: LLMPolicy,
    tokenizer,
    prompts: List[str],
    batch: Dict,
    context_lengths: Sequence[int],
    gen_kwargs: Dict,
):
    """Generate *responses* for each *context length* in ``context_lengths``.

    Returns a mapping ``{context_len: List[str]}`` where the list length is
    ``batch_size``.
    """
    assert policy.cfg.is_conditional, "Policy must be conditional."
    
    device = policy.model.decoder.conditional_llm.device 

    pairs_C, choices_C = batch["pairs_C"].to(device), batch["choices_C"].to(device)

    model_inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding='max_length',
        truncation=True,
        max_length=512
    ).to(device)

    generations = {cl: [] for cl in context_lengths}

    for ctx_len in context_lengths:
        bs = pairs_C.shape[0]
        z_C_large = torch.zeros(bs, pairs_C.shape[1], 256, dtype=torch.float32).to(device)
        
        if ctx_len > 0:
            for b in range(bs):
                for pos in range(pairs_C.shape[1]):
                    # Sample context indices for this input
                    context_idx = torch.tensor(np.random.choice(10, ctx_len, replace=False), dtype=torch.long)
                    
                    # Extract context pairs and choices for this input
                    pairs_C_sliced = pairs_C[b, context_idx].clone().unsqueeze(0)
                    choices_C_sliced = choices_C[b, context_idx].clone().unsqueeze(0)
                    
                    # Get latent variable for each context position
                
                    with torch.no_grad():
                        z_C = policy.model.get_latent_var(None, None, pairs_C_sliced, choices_C_sliced)
                        z_C_large[b, pos] = z_C[0, 0].to(torch.float32)  # Take first batch, first position

        with torch.autocast(torch.device(device).type):
            out_ids = policy.model.decoder.conditional_llm.conditional_generate(
                latent=z_C_large.flatten(0,1),
                input_ids=model_inputs["input_ids"],
                attention_mask=model_inputs["attention_mask"],
                **gen_kwargs,
            )

        decoded = tokenizer.batch_decode(out_ids, skip_special_tokens=True)
        # As above, strip the prompt.
        cleaned = []
        for prompt_text, full in zip(prompts, decoded):
            # Try different possible anchors
            assistant_anchor = "\nassistant\n"
            model_anchor = "\nmodel\n"
            
            # Try to find either anchor
            assistant_idx = full.rfind(assistant_anchor)
            model_idx = full.rfind(model_anchor)
            
            if assistant_idx != -1:
                # Found assistant anchor
                cleaned.append(full[assistant_idx + len(assistant_anchor):].strip())
            elif model_idx != -1:
                # Found model anchor
                cleaned.append(full[model_idx + len(model_anchor):].strip())
            else:
                # Fallback to splitting by prompt
                cleaned.append(full.split(prompt_text, 1)[-1].strip())
        generations[ctx_len] = cleaned
    return generations


def initialize_embedder(device):
        embedder_name = "meta-llama/Meta-Llama-3-8B"
    
        print('Initializing embedder model on', device)
        embedder_model = AutoModel.from_pretrained(
            embedder_name,
            torch_dtype=torch.float16
        ).to(device)
        embedder_model.eval()
        
        reward_tokenizer = AutoTokenizer.from_pretrained(embedder_name)
        if reward_tokenizer.pad_token_id is None:
            reward_tokenizer.pad_token = reward_tokenizer.eos_token
        reward_tokenizer.padding_side = "right"
        
        return embedder_model, reward_tokenizer
    
def get_embedding(model_inputs, model):
    cls_idxs = model_inputs["attention_mask"].sum(1)
    with torch.no_grad():
        last_hidden_state = model(
            **model_inputs,
            output_hidden_states=True,
        ).hidden_states[-1]
        bs = last_hidden_state.shape[0]
        embed = torch.zeros((bs, last_hidden_state.shape[2])).to(model.device)
        for i in range(bs):
            embed[i, :] = last_hidden_state[i, cls_idxs[i], :]

    return embed

################################################################################
# Main script
################################################################################

def set_seed(seed):
    """Set all random seeds
    Args:
        seed (int): integer for reproducible experiments
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True

def main():  # noqa: C901 – a bit long but still readable
    set_seed(42)
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy_save_dir", type=str, required=True)
    parser.add_argument("--help_reward_dir", type=str, required=True)
    parser.add_argument("--honesty_reward_dir", type=str, required=True)
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--context_lengths", type=int, nargs="*", default=[0, 1, 3, 5, 10])
    # Generation hyper‑parameters – feel free to tweak.
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--device", type=int, default=0, help="CUDA device index (0 for single GPU).")
    parser.add_argument("--labels", type=str, default="helpfulness", choices=["helpfulness", "honesty", "truthfulness"])
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    outfile = out_dir / f"{args.split}_generations.jsonl"

    device = torch.device(f"cuda:{args.device}")

    # ---------------------------------------------------------------------
    # Load models
    # ---------------------------------------------------------------------
    policy_cfg_full, policy = load_model_from_save(args.policy_save_dir, LLMPolicy, device)
    _, reward_help = load_model_from_save(args.help_reward_dir, RewardModel, device)
    _, reward_hon = load_model_from_save(args.honesty_reward_dir, RewardModel, device)
    reward_embedder, reward_tokenizer = initialize_embedder(device)

    tokenizer = AutoTokenizer.from_pretrained(policy_cfg_full.model.llm_name)  # type: ignore[attr-defined]
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # Important for generation with causal models.

    # ---------------------------------------------------------------------
    # Load data (uses *policy* cfg to stay in sync with tokenisation etc.)
    # ---------------------------------------------------------------------
    policy_cfg_full.data.labels = [args.labels]
    policy_cfg_full.data.batch_size = 1
    policy_cfg_full.data.num_targets = 15
    dataloaders = setup_dataloaders(policy_cfg_full.data, splits=[args.split])  # type: ignore[attr-defined]
    dataloader = dataloaders[args.split]

    gen_kwargs = dict(
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        do_sample=True,
    )

    
    sample_id = 0
    sample_id_2 = 0
    with outfile.open("w") as f_out:
        for batch in tqdm(dataloader, desc=f"Generating ({args.split})"):
            # Remove all the pairs from the target, as we only need single forward passes
            batch["pairs_T"]["input_ids"] = batch["pairs_T"]["input_ids"][:,:,[0]]
            batch["pairs_T"]["attention_mask"] = batch["pairs_T"]["attention_mask"][:,:,[0]]
            batch["pairs_T"]["cls_idx"] = batch["pairs_T"]["cls_idx"][:,:,[0]]
            batch["pairs_T"]["response_start_idx"] = batch["pairs_T"]["response_start_idx"][:,:,[0]]
            
            # Extract the base prompts from the tokenised batch.
            input_ids = batch["pairs_T"]["input_ids"].view(-1, batch["pairs_T"]["input_ids"].shape[-1])
            response_start_idx = batch["pairs_T"]["response_start_idx"].view(-1)
            seq_length = input_ids.size(1)
            mask = torch.arange(seq_length, device=input_ids.device)[None, :] >= (response_start_idx - 2)[:, None]
            input_ids[mask] = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
            
            # They nasty (full chat template) prompts
            dirty_prompts = tokenizer.batch_decode(input_ids, skip_special_tokens=False, ignore_pad_token=True)
            dirty_prompts = [dp.replace(tokenizer.pad_token, "") for dp in dirty_prompts] # not sure why this is needed, but it is.
            
            # Clean, with no mumbo jumbo around it!
            prompts = [clean_prompt(p) for p in tokenizer.batch_decode(input_ids, skip_special_tokens=True)]
            
            if policy_cfg_full.model.is_conditional:  # type: ignore[attr-defined]
                gens = generate_conditional(
                    policy,
                    tokenizer,
                    dirty_prompts,
                    batch,
                    context_lengths=args.context_lengths,
                    gen_kwargs=gen_kwargs,
                )
            else:
                responses = generate_simple(
                    policy,
                    tokenizer,
                    dirty_prompts,
                    gen_kwargs,
                )
                gens = {0: responses}  # use 0 to denote *no* context.

            # ----------------------------------------------------------------
            # Score with *both* reward models
            # ----------------------------------------------------------------
            for ctx_len, resp_list in gens.items():
                sample_id = sample_id_2
                assert len(resp_list) == len(prompts)
                # Build *joined* texts for reward evaluation.
                joined = [build_chat_prompt_reward(p, r) for p, r in zip(prompts, resp_list)]
                toks = reward_tokenizer(
                    joined,
                    return_tensors="pt",
                    padding='max_length',
                    max_length=1024,
                    truncation=True,
                ).to(device)

                with torch.no_grad():
                    # RewardModel.forward expects the *target* pair structure.  We can
                    # use the simple variant (pairs_T only) with *choices_T* all zeros
                    # because we are interested in the *scalar* reward, not a
                    # comparison between A/B.  The TargetEncoder consumes ``pairs_T``
                    # shaped (bs, num_targets, 2, input_dim).  Here we set
                    # ``num_targets=1`` and duplicate the response across the A/B
                    # dimension so the *difference* is zero and hence ``logp_choices``
                    # will use only the raw *reward*.
                    
                    reward_embedding = get_embedding(toks, reward_embedder) # TODO
                    reward_embedding = reward_embedding.unsqueeze(1).repeat(1, 1, 2, 1)
                    choices_zeros = torch.zeros(1, len(joined), 1, dtype=torch.long, device=device)

                    help_out = reward_help(reward_embedding, choices_zeros)
                    honesty_out = reward_hon(reward_embedding, choices_zeros)

                    # Reward is ``rewards.squeeze(-1)`` inside the output dict.
                    help_scores = help_out["rewards"].squeeze().mean(dim=-1).tolist()
                    honesty_scores = honesty_out["rewards"].squeeze().mean(dim=-1).tolist()

                for prompt_text, resp, help, honest in zip(
                    prompts, resp_list, help_scores, honesty_scores  # type: ignore[index]
                ):
                    record = {
                        "id": sample_id,
                        "context_len": ctx_len,
                        "helpfulness_score": help,
                        "honesty_score": honest,
                        "prompt": prompt_text,
                        "response": resp,
                    }
                    f_out.write(json.dumps(record, ensure_ascii=False) + "\n")
                    
                    sample_id += 1
                    
            sample_id_2 = sample_id
                    
            if sample_id > 500:  # Limit the number of samples to avoid too large output files
                print(f"Reached sample limit of 500, stopping generation.")
                break
        

    print(f"Saved generations to {outfile.resolve()}")


if __name__ == "__main__":
    main()
