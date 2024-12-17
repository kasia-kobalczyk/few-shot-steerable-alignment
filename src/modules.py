import torch
import torch.nn as nn
import os 
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.utils import setup_llm

class MLP(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        num_hidden,
        output_size,
        activation=nn.GELU(),
        batch_norm=False,
    ):
        super(MLP, self).__init__()

        self.layers = nn.ModuleList()

        if num_hidden > 0:
            for i in range(num_hidden - 1):
                self.layers.append(nn.Linear(input_size, hidden_size))
                if batch_norm:
                    self.layers.append(nn.BatchNorm1d(hidden_size))
                self.layers.append(activation)
                input_size = hidden_size

            self.layers.append(nn.Linear(hidden_size, output_size, bias=not batch_norm))

        else:
            self.layers.append(nn.Linear(input_size, output_size))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class FiLM(nn.Module):
    """
    FiLM layer with an MLP for gamma and beta, configurable by the number of layers and intermediate hidden size.
    """

    def __init__(self, input_size, hidden_size, output_size, num_hidden=2):
        super(FiLM, self).__init__()

        # Create the MLP for gamma and beta with the specified number of layers and hidden dimensions
        self.gamma_mlp = MLP(
            input_size=input_size, 
            hidden_size=hidden_size, 
            output_size=output_size, 
            num_hidden=num_hidden
        )
        self.beta_mlp = MLP(
            input_size=input_size, 
            hidden_size=hidden_size, 
            output_size=output_size, 
            num_hidden=num_hidden
        )

        self._initialize_last_gamma_weights(self.gamma_mlp.layers)
        self._initialize_last_beta_weights(self.beta_mlp.layers)

    def _initialize_last_gamma_weights(self, mlp):
        """
        Initialize MLP weights with small values for near-identity behavior.
        """
        nn.init.normal_(mlp[-1].weight, mean=0.0, std=1e-4)
        nn.init.normal_(mlp[-1].bias, mean=1.0, std=1e-4)

    def _initialize_last_beta_weights(self, mlp):
        """
        Initialize MLP weights with small values for near-identity behavior.
        """
        nn.init.normal_(mlp[-1].weight, mean=0.0, std=1e-4)
        nn.init.normal_(mlp[-1].bias, mean=0.0, std=1e-4)

    def forward(self, hidden_states_tuple, latent_variable):
        # Apply FiLM modulation: gamma * hidden_states + beta
        gamma = self.gamma_mlp(latent_variable).unsqueeze(1)
        beta = self.beta_mlp(latent_variable).unsqueeze(1)
        return (gamma * hidden_states_tuple[0] + beta,) + hidden_states_tuple[1:]


class ConditionalLLM(nn.Module):
    """
    AutoModelForCausalLM model with FiLM layers for conditional modulation.
    """

    def __init__(
            self, 
            llm_name, 
            num_film_layers, 
            is_lora_tunable, 
            lora_kwargs, 
            hidden_dim
        ):
        super(ConditionalLLM, self).__init__()


        # Store the base model
        self.base_model = setup_llm(
            llm_name,
            causal=True,
            is_lora_tunable=is_lora_tunable,
            lora_kwargs=lora_kwargs,
        )

        self.layers = self._get_transformer_layers()
        if num_film_layers == -1:
            num_film_layers = len(self.layers)

        self.num_film_layers = num_film_layers
        
    
        # Add FiLM layers
        print('Setting up FiLM layers ...')
        self.film_layers = nn.ModuleList(
            [
                FiLM(
                    input_size=hidden_dim, 
                    hidden_size=hidden_dim, 
                    output_size=self.base_model.config.hidden_size,
                    num_hidden=1,
                )
                for _ in range(self.num_film_layers)
            ]
        )
        print('Registering FiLM layers ...')
        self._register_film_hooks()

    def _apply_film(self, module, input, output, film_layer, latent_variable):
        return film_layer(output, latent_variable)

    def _register_film_hooks(self):
        for i, layer_module in enumerate(self.layers[-self.num_film_layers:]):
            layer_module.register_forward_hook(
                lambda module, input, output, i=i: self._apply_film(
                    module,
                    input,
                    output,
                    self.film_layers[i],
                    self.current_latent_variable,
                )
            )

    def _get_transformer_layers(self):
        if hasattr(self.base_model, "transformer"):
            return self.base_model.transformer.h
        elif hasattr(self.base_model, "encoder"):
            return self.base_model.encoder.layer
        elif hasattr(self.base_model, "model"):
            try:
                return self.base_model.model.layers
            except AttributeError:
                return self.base_model.model.model.layers
        elif hasattr(self.base_model, "layers"):
            return self.base_model.layers
        else:
            raise ValueError("Unsupported transformer architecture")

    def forward(self, input_ids, attention_mask=None, latent_variable=None):
        self.current_latent_variable = latent_variable
        with torch.amp.autocast(device_type='cuda'):
            outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
        return outputs

    def conditional_generate(
        self,
        latent_variable,
        input_ids,
        **kwargs
    ):

        # Set the latent variable on the hook and go fishing!
        self.current_latent_variable = latent_variable

        generated_ids = self.base_model.generate(
            input_ids,
            **kwargs
        )

        return generated_ids

    def __getattr__(self, name):
        """
        Delegate attribute access to the base model if not found in this class.
        This allows access to all methods and attributes of the base model.
        """
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.base_model, name)