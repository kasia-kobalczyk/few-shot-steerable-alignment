import torch
import torch.nn as nn
import os 
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.utils import setup_llm

import torch
import torch.nn as nn
from typing import List, Sequence, Optional, Tuple, Union

# ------------------------------------------------------------
# 1. Small reusable MLP helper
# ------------------------------------------------------------
class MLP(nn.Module):
    """Simple feed‑forward network: (Linear→BN?→Act)×(num_hidden‑1) → Linear."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        num_hidden: int = 1,
        activation: nn.Module = nn.GELU(),
        batch_norm: bool = False,
    ) -> None:
        super().__init__()

        layers: List[nn.Module] = []
        dim_in = input_size
        for _ in range(max(num_hidden - 1, 0)):
            layers.append(nn.Linear(dim_in, hidden_size, bias=not batch_norm))
            if batch_norm:
                layers.append(nn.BatchNorm1d(hidden_size))
            layers.append(activation)
            dim_in = hidden_size
        # last linear
        layers.append(nn.Linear(dim_in, output_size, bias=not batch_norm))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, *)
        return self.net(x)

# ------------------------------------------------------------
# 2. FiLM modulation block (Feature‑wise Linear Modulation)
# ------------------------------------------------------------
class FiLM(nn.Module):
    """Modulates hidden states given a latent tensor γ, β = f(latent)."""

    def __init__(
        self,
        latent_dim: int,
        hidden_dim: int,
        target_dim: int,
        num_hidden: int = 1,
    ) -> None:
        super().__init__()
        self.gamma_mlp = MLP(latent_dim, hidden_dim, target_dim, num_hidden=num_hidden)
        self.beta_mlp = MLP(latent_dim, hidden_dim, target_dim, num_hidden=num_hidden)
        self._init_identity()

    def _init_identity(self) -> None:
        """Near‑identity init so base model starts untouched."""
        nn.init.zeros_(self.gamma_mlp.net[-1].weight)   # multiplicative scale
        nn.init.ones_ (self.gamma_mlp.net[-1].bias)

        nn.init.zeros_(self.beta_mlp.net[-1].weight)    # additive shift
        nn.init.zeros_(self.beta_mlp.net[-1].bias)  

    def forward(self, hidden: torch.Tensor, latent: torch.Tensor) -> torch.Tensor:  # (B,L,H)
        multiplier = self.gamma_mlp(latent).unsqueeze(1)  # (B,1,H)
        bias = self.beta_mlp(latent).unsqueeze(1)
        return hidden * multiplier + bias

# ------------------------------------------------------------
# 3. Wrapper block that embeds FiLM directly (no hooks, batch‑safe)
# ------------------------------------------------------------
class BlockWithFiLM(nn.Module):
    """Replaces a transformer block such that FiLM is *inside* the forward graph."""

    def __init__(self, base_block: nn.Module, film: FiLM):
        super().__init__()
        self.block = base_block
        self.film = film

    def forward(
        self,
        hidden_states: torch.Tensor,
        latent: torch.Tensor = None,
        **kwargs,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        """Assumes the underlying block returns hidden_states as first element."""
        if latent is None:
            return self.block(hidden_states, **kwargs)
        out = self.block(hidden_states, **kwargs)
        if isinstance(out, tuple):
            hidden, *rest = out
            hidden = self.film(hidden, latent)
            return (hidden, *rest)
        else:
            return self.film(out, latent)
        
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.block, name)

# ------------------------------------------------------------
# 4. ConditionalLLM – no forward hooks, supports per‑sample latents
# ------------------------------------------------------------
class ConditionalLLM(nn.Module):
    """LLM wrapper that injects FiLM modulations inside chosen transformer layers."""

    def __init__(
        self,
        llm_name: str,
        num_film_layers: Union[int, Sequence[int]] = -1,
        hidden_dim: int = 512,
        is_lora_tunable: bool = False,
        lora_kwargs: Optional[dict] = None,
    ) -> None:
        super().__init__()

        # 4.1 Load the base causal LM (HuggingFace GPT‑like)
        self.base = setup_llm(
            llm_name,
            causal=True,
            is_lora_tunable=is_lora_tunable,
            lora_kwargs=lora_kwargs or {},
        )

        # 4.2 Get the list of transformer layers (robust across models)
        self.layers = self._get_layers()
        hidden_size = self.base.config.hidden_size

        # 4.3 Decide which layers to FiLM
        if isinstance(num_film_layers, int):  # e.g. 12  or  -1 for all
            if num_film_layers == -1:
                indices = list(range(len(self.layers)))
            else:
                indices = list(range(num_film_layers))
        else:
            indices = list(num_film_layers)
        self.film_indices = indices

        # 4.4 Replace chosen layers with FiLM‑wrapped versions
        for idx in indices:
            orig = self.layers[idx]
            film = FiLM(hidden_dim, hidden_dim, hidden_size, num_hidden=1)
            self.layers[idx] = BlockWithFiLM(orig, film)

    # --------------------------------------------------------
    # forward / generate
    # --------------------------------------------------------
    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None, latent_variable: Optional[torch.Tensor] = None, **kwargs):
        return self.base(input_ids=input_ids, attention_mask=attention_mask, latent=latent_variable, **kwargs)

    @torch.no_grad()
    def conditional_generate(self, latent: torch.Tensor, input_ids: torch.Tensor, **generate_kwargs):
        
        def _add_latent(module, args, kwargs):
            kwargs["latent"] = latent
            return args, kwargs

        # Register hook for each transformer layer that has FiLM
        handles = []
        for idx in self.film_indices:
            handle = self.layers[idx].register_forward_pre_hook(_add_latent, with_kwargs=True)
            handles.append(handle)
            
        try:
            result = self.base.generate(input_ids, **generate_kwargs)
            return result
        finally:
            for handle in handles:
                handle.remove()

    # --------------------------------------------------------
    # helpers
    # --------------------------------------------------------
    def _get_layers(self):
        """Return list‑like container of transformer blocks."""
        model = self.base
        if hasattr(model, "transformer"):
            return model.transformer.h
        if hasattr(model, "encoder"):
            return model.encoder.layer
        if hasattr(model, "model"):
            try:
                return model.model.layers
            except AttributeError:
                return model.model.model.layers
        if hasattr(model, "layers"):
            return model.layers
        raise ValueError("Unsupported architecture: cannot find transformer layers")

    # ------------------------------------------------------------------
    # Attribute passthrough: anything not found here → delegate to base.
    # ------------------------------------------------------------------
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.base, name)
