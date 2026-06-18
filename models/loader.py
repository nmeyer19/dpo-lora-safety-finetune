import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import PreTrainedTokenizer, PreTrainedModel
from peft import PeftModel

def load_model(
    config: dict, 
    checkpoint_path: str | None = None
) -> tuple[PreTrainedModel, PreTrainedTokenizer]:
    
    cfg = config["model"]
    tokenizer = AutoTokenizer.from_pretrained(cfg["name"])
    
    # no padding by default w llama - need to add
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    base_model = AutoModelForCausalLM.from_pretrained(
        cfg["name"], 
        dtype=getattr(torch, cfg["dtype"]),
    )

    # resize token embedding dimension to account for the new pad token
    base_model.resize_token_embeddings(len(tokenizer))

    if checkpoint_path:
        lora_model = PeftModel.from_pretrained(base_model, checkpoint_path)
        merged_lora = lora_model.merge_and_unload()
        return (merged_lora, tokenizer)
    else:
        return (base_model, tokenizer)