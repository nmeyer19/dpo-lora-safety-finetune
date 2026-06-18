# Overview


## SFT Training
My initial learning rate for SFT was 2e-5. Finetuning on 5k examples from the dolly dataset, the loss curve really didn't seem to move much (looking at the smoothed exponential moving average). That said, I waited to run the full training and actually evaluate on the MMLU dataset. Instead of going straight into the DPO file / training loop, I decided to write up the MMLU eval code first and test it on the base and SFT models. That way, we can either first adjust the SFT pipeline as necessary, or if it actually did meaningfully improve capabilities, we could just save these results and then test on the SFT + DPO model afterwards.

I ran into an issue with the MMLU eval—I thought it could be the way we were tokeninzing the responses or getting the token ID's for the relevant response, but it happened to just be a simple (simple, yet it kinda took me too long to figure out this was the issue) type mismatch where the dataset holds the responses as index numbers of the choices (0-3) but I wasn't converting≤ our models' answers from the letters back to these numbers, so naturally they were both scoring 0% across the board,,.

## MMLU Evaluation
Instead of parsing responses, I decided to go with just argmax-ing the log probabilities of the specific responses " A", " B", " C", and " D". This is much simpler in practice, and actually the more standard approach. The intuition here is that while these token probabilities don't represent the full spectrum of possible answers, their relative ordering indicate which the model is most confident in in general. 

When we do a forward pass on the evaluation prompt, the output is a tensor of shape `[batch_size, seq_length, vocab_size]`. 
- We then have to index into the tensor for each element in `batch_size` (i.e. every example in the batch). 
- Then for each example, we look at the position $i$ in `seq_length`, where $i$ is the last token in the sequence where our attention mask is 1 (i.e. not a padding token). 
- The logits in `vocab_size` at this position represent the model's distribution over what token comes after this position (which is roughly "Answer:") for this example. We'll look up the token IDs for " A", " B", " C", and " D" in this vector and return the argmax of these logits, which is our model's answer.