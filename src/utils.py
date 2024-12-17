from torch.distributions import Normal, Categorical
from torch.distributions.independent import Independent
import torch
from transformers import AutoModel, AutoModelForCausalLM
from peft import LoraModel, LoraConfig



def MultivariateNormalDiag(loc, scale_diag):
    """Multi variate Gaussian with a diagonal covariance function (on the last dimension)."""
    if loc.dim() < 1:
        raise ValueError("loc must be at least one-dimensional.")
    return Independent(Normal(loc, scale_diag), 1)


def IndependentMultinomial(logits):
    """Multinomial distribution with independent trials."""
    if logits.dim() < 1:
        raise ValueError("logits must be at least one-dimensional.")
    return Independent(Categorical(logits=logits), 1)

def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )

def setup_llm(
        llm_name, 
        causal=False,
        is_lora_tunable=False,
        lora_kwargs=None,
    ):
    print('Setting up base LLM: ', llm_name)
    if not causal:
        model = AutoModel.from_pretrained(
            llm_name,
            torch_dtype=torch.float16,
            output_hidden_states=True
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            llm_name,
            torch_dtype=torch.float16,
            output_hidden_states=True,
            is_decoder=True
        )
    if not is_lora_tunable:
        for param in model.parameters():
            param.requires_grad = False
    else:
        print("Setting up LORA model ...")
        peft_config = LoraConfig(
            lora_alpha=lora_kwargs['alpha'],
            lora_dropout=lora_kwargs['dropout'],
            r=lora_kwargs['r'],
            target_modules=["q_proj", "v_proj"],
            task_type="CAUSAL_LM",
            bias="none",
        )
        model = LoraModel(model, peft_config, "default")
        print_trainable_parameters(model)
    return model