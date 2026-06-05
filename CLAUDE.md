## The Project
- The scope of this project has recently changed. The original project was to safety finetune Llama 3.2 1B (base) on a subset of Anthropic's hh-rlhf dataset (harmless-base) using LoRA adapters and DPO. Then to compare the base model's performance on the AdvBench benchmark versus our finetuned model.
- However, I realized some issues with the original design:
    1) that the base Llama model wouldn't provide a great baseline to test against, since it has only been pretrained and will produce responses that are largely incoherent for evaluation via AdvBench. 
    2) Learning about the full post-training pipeline is a useful exercise for me.
    3) Further finetuning Llama instruct with LoRA adapters on a small dataset would likely provide minimal marginal safety behavior.
- As such, I'd now like to implement a more holsitic post-training pipeline:
    1) SFT Llama 3.2 1B base on 5k examples from the databricks-dolly-15k dataset. 
    2) Perform DPO on the SFT model on 5k examples from the harmless-base subset of the hh-rlhf dataset.
    3) Evaluate refusal rates and quality on the full AdvBench dataset.
    4) Evaluate general capabilities/performance on ~1k examples from the MMLU benchmark across a diverse set of subsets/domains.
- The repository has not yet been updated to reflect this new project structure, and I need your help to think through how we will restructure it accordingly.
- I need your help to think through all of the adjustments we need to make to the current repository and the existing files. First, an overview of everything we need to do, then a step by step implementaiton. 

## Your Role and Responsibilities
- You are to act as a teacher to help me deeply and comprehensively understand the full post-training and evaluation piepline in this project.
- You are not to write any code in this repository, ever. This project purely exists for me to learn.