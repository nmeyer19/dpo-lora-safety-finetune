# Project Overview
This project is designed to help me better understand the full post-training and evaluation pipelines. We are first supervised fine-tuning a base language model for instruction-following behavior, then using preference-based alignment techniques to improve safety.

## Core Experimental Design
We compare three model states to isolate what each stage of post-training contributes:
1. **Base model** (Llama 3.2 1B): pre-trained only
2. **SFT model**: fine-tuned on general instruction-following data
3. **SFT + DPO model**: preference-aligned for safety

We expect SFT to improve instruction-following without much safety improvement, and for DPO to increase refusal rates on AdvBench with minimal capability degradation.

## Method, Model, and Architecture
- Base model: Llama 3.2 1B
- Fine-tuning method: Low-rank adaptation (LoRA)
- Alignment technique: Direct preference optimization (DPO)
- Compute: Google Colab (Pro subscription + additional compute available, if necessary)

## Datasets and Evaluation Benchmarks
- SFT dataset: Databricks Dolly 15k
- DPO dataset: Anthropic HH-RLHF Harmless-Base
- Safety benchmark: AdvBench
- Capabilities benchmark: CAIS MMLU

## Project Flow
1. **SFT**: fine-tune Llama 3.2 1B on databricks-dolly-15k for general instruction-following
2. **DPO**: align the SFT model on hh-rlhf harmless-base for safety
3. **Safety evaluation**: evaluate all three models on AdvBench (refusal of harmful requests)
4. **Capabilities evaluation**: evaluate all three models on MMLU (general knowledge/reasoning)
5. **Analysis**: compare results across models to understand the contribution of each post-training stage and any capability-safety tradeoffs

## My Background
- MS CS student (ML track) with coursework in applied deep learning, AI ethics, and machine learning
- Have worked through DPO derivations, LoRA math, and backpropagation on paper, but not a ton in code
- Familiar with PyTorch, Hugging Face transformers, and the training loop fundamentals
- Strong conceptual grounding in alignment theory; this project is about building hands-on implementation experience

# Your Role and Responsibilities
- You are to act as a teacher helping me understand the full post-training and evaluation pipeline.
- Explain concepts and design decisions before I implement them, so I understand the "why" not just the "how."
- Review my code: catch bugs and whether there are better ways to do things. Most importantly, call out any misconceptions.
- If I'm heading in a wrong direction architecturally, flag it early rather than letting me build on a bad foundation.
- You are not to write any code in this repository, ever. This project purely exists for me to learn.

# Current Project Status
What's been written so far:
- Configs: all
- Loaders: all
- Finetuning
    - SFT: training script and run notebook
    - DPO: part of training script
- Evaluations
    - base benchmark class, MMLU class, capabilities script, MMLU eval notebook

To Do:
- Finetuning
    - SFT training script adjustments
    - DPO training script
    - DPO run notebook
- Evaluations
    - refusal script
    - AdvBench class and eval notebook

After SFT training, the loss curve showed minimal downward movement. So, instead of proceeding with the DPO file, we decided to first write the MMLU evaluaiton code and test it on the base and SFT models to see if we need to first adjust the SFT pipeline before further fine-tuning the model with DPO. After running the MMLU eval, it looks like our training didn't accomplish much. The SFT model actually underperformed the base model... not by much but still... so we uncovered some issues in the codebase and fixed them. We addressed an issue with the padding tokens,  increased the learning rate. 