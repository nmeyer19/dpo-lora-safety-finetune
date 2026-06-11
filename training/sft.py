import yaml
from data.loaders.dolly import DollyDataLoader
from transformers import get_linear_schedule_with_warmup, DataCollatorForSeq2Seq
from models.loader import load_model
from peft import get_peft_model, LoraConfig, TaskType
import torch
from torch.utils.data import DataLoader

# load the config
with open("./configs/sft.yaml", "r") as file:
    config = yaml.safe_load(file)

# load training data
dataloader = DollyDataLoader(config)
dataloader.load()
dataset = dataloader.get_data()

# load base model and tokenizer
base_model, tokenizer = load_model(config)

# tokenizes an example and masks the prompt tokens
def tokenize(example):
    # tokenize full example text
    example_text = example["prompt"] + example["response"]
    tokenized_ex = tokenizer(example_text, truncation=True, 
                             max_length=config["model"]["max_length"],
                             padding=False)
    # mask prompt tokens for no gradient signal
    prompt_len = len(tokenizer(example["prompt"], truncation=True,
                     max_length=config["model"]["max_length"],
                     padding=False)["input_ids"])
    labels = tokenized_ex["input_ids"].copy()
    for i in range(prompt_len):
        labels[i] = -100
    # create a new field indicating which tokens to ignore in gradient calcs
    tokenized_ex["labels"] = labels 
    
    return tokenized_ex

tokenized_dataset = dataset.map(tokenize, remove_columns=dataset.column_names)

# construct the lora model
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=config["lora"]["r"],
    target_modules=config["lora"]["target_modules"],
    lora_alpha=config["lora"]["alpha"],
    lora_dropout=config["lora"]["dropout"],
)
lora_model = get_peft_model(base_model, lora_config)
# trainable params: 1,703,936 || all params: 1,237,518,336 || trainable%: 0.1377

# optimizer: AdamW
optimizer = torch.optim.AdamW(lora_model.parameters(), 
                              lr=config["training"]["learning_rate"])

# HF Seq2Seq collator for masked padding
collator = DataCollatorForSeq2Seq(tokenizer, lora_model, padding=True, 
                                  label_pad_token_id=-100)

# torch dataloader for training
train_dataloader = DataLoader(tokenized_dataset, 
                              batch_size=config["training"]["batch_size"],
                              shuffle=True, collate_fn=collator)

total_steps = ((len(train_dataloader) // 
               config["training"]["gradient_accumulation_steps"]) * 
               config["training"]["num_epochs"])

# lr scheduler
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=config["training"]["warmup_steps"],
    num_training_steps=total_steps)

# put the model in training mode
lora_model.train()

# training loop



"""
4b. write the training loop itself (foward pass, loss, backward pass, 
optimizer.step)
- for each batch, loss = forward_pass(batch) / gradient_accumulation_steps
before calling loss.backward()
- only if step % gradient_accumulation_steps == 0 then optimizer.step() and
optimizer.zero_grad()
4c. within the training loop, log to wandb                  
5. save final lora adapter matrices/weights 
- model.save_pretrained(output_dir) - saves only the adapter matrices and the
lora config
"""