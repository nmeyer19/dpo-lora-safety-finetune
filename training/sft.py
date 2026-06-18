import yaml
from data.loaders.dolly import DollyDataLoader
from transformers import get_linear_schedule_with_warmup, DataCollatorForSeq2Seq
from models.loader import load_model
from peft import get_peft_model, LoraConfig, TaskType
import torch
from torch.utils.data import DataLoader
import wandb 

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
    # tokenize prompt and response separately
    tokenized_prompt = tokenizer(example["prompt"], padding=False)
    
    tokenized_response = tokenizer(example["response"], padding=False, 
                                   add_special_tokens=False)

    # concatenate token ids and truncate to max_length config
    prompt_ids = tokenized_prompt["input_ids"] 
    response_ids = tokenized_response["input_ids"]
    input_ids = prompt_ids + response_ids
    input_ids = input_ids[:config["model"]["max_length"]]

    # mask prompt tokens for no gradient signal and truncate length
    # and truncate to max_length config
    labels = [-100]*len(prompt_ids) + response_ids
    labels = labels[:config["model"]["max_length"]]

    # attention_mask all 1s (inherits truncation from labels)
    attention_mask = [1] * len(input_ids)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }

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

# train mode
lora_model.train()

# training-relevant initializations
acc_loss = 0
step = 1

total_epochs = config["training"]["num_epochs"]
gradient_accumulations = config["training"]["gradient_accumulation_steps"]
logging_steps = config["training"]["logging_steps"]

wandb.init(project=config["wandb"]["project"],
           name=config["wandb"]["run_name"],
           config=config)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
lora_model = lora_model.to(device)

# just in case
optimizer.zero_grad()

# training loop
for epoch in range(total_epochs):
    for batch in train_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()} # move batch to GPU
        outputs = lora_model(**batch)                       # forward pass
        loss = outputs.loss / gradient_accumulations        # normalized loss
        acc_loss += loss.item()                             # accumulated loss
        loss.backward()                                     # backward pass
                                                            # accumulates gradients

        if step % gradient_accumulations == 0:
            optimizer.step()                                # update weights
            optimizer.zero_grad()                           # zero out accumulated gradients
            scheduler.step()                                # update LR
            
            if step % (gradient_accumulations * logging_steps) == 0:
                wandb.log({"loss": acc_loss / logging_steps, 
                           "lr": scheduler.get_last_lr()[0]})
                acc_loss = 0                                # reset accumulation
        
        step += 1

# save everything
lora_model.save_pretrained(config["outputs"]["model_dir"])
tokenizer.save_pretrained(config["outputs"]["model_dir"])
wandb.finish()


"""
Logging bug
acc_loss logging -> it's reset after every optimizer step (as it should be)
but only logged every 10 steps, so each logging isn't an accurate representation
of the loss being accrued over this horizon, only the last step.

General improvement
More LoRA target modules (k_proj, o_proj) - but will increase param count
and training intensity
"""