# Project Overview
This project is designed to help me better understand the full post-training and evaluation pipelines. We are first supervised fine-tuning a base language model for instruction-following behavior, then using preference-based alignment techniques to improve safety.

## Core Experimental Design
We compare three model states to isolate what each stage of post-training contributes:
1. **Base model** (Llama 3.2 1B): pre-trained only
2. **SFT model**: fine-tuned on general instruction-following data
3. **SFT + DPO model**: preference-aligned for safety

We expect SFT to improve instruction-following without much safety improvement, and for DPO to increase refusal rates on AdvBench with minimal capability degradation. We treat the absolute scores—especially for a 1B model with LoRA SFT on 15k examples—as secondary; the **deltas between stages** are the primary signal.

## Method, Model, and Architecture
- Base model: Llama 3.2 1B
- Fine-tuning method: Low-rank adaptation (LoRA), with separate adapter sets for the SFT and DPO stages
- Alignment technique: Direct preference optimization (DPO)
- Compute: Google Colab (Pro subscription + additional compute available, if necessary)

## Datasets and Evaluation Benchmarks
- SFT dataset: Databricks Dolly 15k
- DPO dataset: Anthropic HH-RLHF Harmless-Base
- Safety benchmark: AdvBench
- Capabilities benchmark: CAIS MMLU
- Instruction-following benchmark: IFEval
- SFT training-success metric: held-out Dolly cross-entropy loss

## Evaluation Structure
Each metric is scoped to what it itself can actually show:
**Did SFT fit the target distribution?**
- Eval: cross-entropy loss on a held-out validation set of the dolly dataset
- Models Evaluated: SFT, SFT+DPO
**Did instruction-following *generalize* to unseen prompts (the real SFT goal) and did DPO induce over-refusal?**
- Eval: IFEval
- Models Evaluated: all
**General Capability *conditional on format compliance***
- Eval: MMLU
- Models Evaluated: all
**Refusal of harmful requests (DPO safety gain)**
- Eval: AdvBench
- Models Evaluated: All

Key scoping notes:
- **Held-out Dolly loss** is a within-SFT-lineage diagnostic. It is *not* run on the base model (which has never seen the Dolly format, so its loss would be high for reasons unrelated to capability) and it is necessary-but-not-sufficient: it confirms the adapters fit Dolly, not that the model is a good assistant in general.
- **IFEval** is where we claim generalization, because it is verifiable (programmatic constraint checks, no judge model or hand-grading). If SFT+DPO drops materially below SFT here, that surfaces the helpfulness/over-refusal tax from harmless-base DPO—so a separate helpfulness eval is not needed.
- **MMLU** is measured under format-matched conditions (see below); it reflects capability given that each model receives the prompt format it was trained to expect.

## MMLU Evaluation Protocol
MMLU scores are highly sensitive to prompting, scoring, and tokenization choices, so the protocol is fixed across all three models:
- **Scoring:** log-likelihood over the answer-letter tokens `" A" / " B" / " C" / " D"` (with leading space), argmax. The prompt ends without a trailing space so the space lives on the answer token; token IDs are verified on sample prompts to rule out double-spacing.
- **Few-shot:** 5-shot for all models, identical exemplars, with exemplars wrapped in the same scaffold the model receives.
- **Format matching:** the base model receives a plain MMLU prompt; the SFT and SFT+DPO models receive the *same* question and choices wrapped in the Dolly instruction template (`### Instruction: ... ### Response:`). Only the surrounding scaffold changes—the question content is held constant—so capability is isolated from format adherence.
- **Cross-check:** the from-scratch implementation is validated against `lm-evaluation-harness` on the same three models. Agreement within ~1% validates the custom eval code (and by extension confidence in the AdvBench/Dolly-loss code); disagreement localizes an implementation gap.

## Project Flow
1. **SFT**: fine-tune Llama 3.2 1B on databricks-dolly-15k for general instruction-following
2. **DPO**: align the SFT model on hh-rlhf harmless-base for safety
3. **Training-success check**: evaluate SFT and SFT+DPO on held-out Dolly loss (gate before downstream evals)
4. **Instruction-following evaluation**: evaluate all three models on IFEval
5. **Capabilities evaluation**: evaluate all three models on MMLU (format-matched, 5-shot, log-likelihood)
6. **Safety evaluation**: evaluate all three models on AdvBench (refusal of harmful requests)
7. **Analysis**: compare results across models to understand the contribution of each post-training stage and any capability-safety tradeoffs

## Evaluation Design Rationale
The evaluation structure evolved during the project after the initial MMLU + AdvBench plan produced a result that didn't fit the original hypothesis.

**The triggering observation.** Under the original plan (zero-shot MMLU, raw concatenation of question/context/choices, no instruction template), the **base model outperformed the SFT model on MMLU by ~2.5%**. The expectation had been the opposite. Diagnosing this surfaced two distinct problems:

1. **MMLU + AdvBench don't paint the full picture of the SFT stage.** Neither benchmark directly measures the actual goal of SFT—producing assistant-like, instruction-following behavior. MMLU measures knowledge (which SFT isn't really trying to add), and AdvBench measures refusal (which is the DPO stage's job). The SFT stage had no eval that targeted what it was for.

2. **MMLU as originally run conflated capability with format compliance.** SFT on Dolly induces an expected prompt-response format. Feeding the SFT model a raw MMLU prompt evaluates it out-of-distribution at the *format* level, making its answer-letter log-probs noisier. The 2.5% gap is therefore plausibly a prompting/scoring artifact rather than genuine capability loss—consistent with the fact that LoRA freezes the base weights and is low-rank, so a large true capability drop from 15k examples would itself be surprising.

**The resulting changes.**
- **Added held-out Dolly loss** as an explicit SFT training-success gate—cheap, fast, and it catches the embarrassing failure modes (loss didn't drop, adapter not attached). But because it only measures fit to Dolly's specific distribution, it can't stand in for general assistant behavior.
- **Added IFEval** to measure whether instruction-following *generalizes* beyond Dolly to unseen prompts—the actual SFT goal. Its verifiable, programmatic scoring keeps it tractable for a learning project, and running it across all three models lets it double as the DPO over-refusal regression check.
- **Reframed MMLU** from "capability" to "capability conditional on format," and fixed the protocol: log-likelihood letter-scoring, 5-shot (which format-primes the base model and is expected to lift the base side rather than the SFT side), and per-model format matching via the Dolly template. The prediction is that, once format-matched, the base-vs-SFT gap shrinks toward zero or flips slightly; if a meaningful gap survives, *that* is a genuine, reportable finding about SFT trading capability for format adherence.
- **Decided to validate the from-scratch eval against lm-evaluation-harness** rather than replace it. The from-scratch implementation is the point of a learning project; the harness provides comparability to published numbers and a bug cross-check. Agreement validates the custom code; disagreement is itself the most educational outcome.

The throughline: each metric is now framed as what it actually measures (MMLU = capability-given-format, Dolly loss = SFT fit, IFEval = instruction-following generalization, AdvBench = refusal), rather than being forced onto clean capability/alignment/safety axes that the benchmarks don't cleanly occupy.

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
    - MMLU rework: 5-shot, log-likelihood letter-scoring, Dolly-template format matching for SFT/SFT+DPO
    - lm-evaluation-harness cross-check (custom MMLU task YAML with Dolly `doc_to_text` wrapper)
    - held-out Dolly loss eval (SFT, SFT+DPO)
    - IFEval class and eval notebook (all three models)
    - refusal script
    - AdvBench class and eval notebook

# Your Role and Responsibilities
- You are to act as a teacher helping me understand the full post-training and evaluation pipeline.
- Explain concepts and design decisions before I implement them, so I understand the "why" not just the "how."
- Review my code: catch bugs and whether there are better ways to do things. Most importantly, call out any misconceptions.
- If I'm heading in a wrong direction architecturally, flag it early rather than letting me build on a bad foundation.
- You are not to write any code in this repository, ever. This project purely exists for me to learn.