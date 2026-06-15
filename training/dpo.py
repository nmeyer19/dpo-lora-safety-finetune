import yaml
from data.loaders.hh_rlhf import HHRLHFDataLoader
from transformers import get_linear_schedule_with_warmup
from models.loader import load_model
from peft import get_peft_model, LoraConfig, TaskType
import torch
from torch.utils.data import DataLoader
import wandb 

# load the config
with open("./configs/dpo.yaml", "r") as file:
    config = yaml.safe_load(file)

# load training data
dataloader = HHRLHFDataLoader(config)
dataloader.load()
dataset = dataloader.get_data()

# load reference model and tokenizer
reference_model, tokenizer = load_model(config, checkpoint_path=
                                        config["model"]["checkpoint_path"])
reference_model.eval()      # set to eval mode to explicit freeze

# load another instantiation of SFT model as the base for DPO training
base_model, _ = load_model(config, checkpoint_path=
                           config["model"]["checkpoint_path"])

# construct the lora / policy model
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=config["lora"]["r"],
    target_modules=config["lora"]["target_modules"],
    lora_alpha=config["lora"]["alpha"],
    lora_dropout=config["lora"]["dropout"],
)
policy_model = get_peft_model(base_model, lora_config)

# tokenizes an example and masks the prompt tokens
