from data.loaders.mmlu import MMLUDataLoader
from evaluation.benchmarks.base import BaseBenchmark
import torch

class MMLUBenchmark(BaseBenchmark):
    """
    MMLU benchmark object.
    Implements load() and evaluate() functions for MMLU.
    Inherits run() and save_results() implementations from BaseBenchmark ABC.
    Runs the full MMLU eval pipeline through saving results via the run() 
    function call.
    """

    def load(self) -> None: 
        dataloader = MMLUDataLoader(self.config)
        dataloader.load()
        self.data = dataloader.get_data()

    def evaluate(self) -> None:
        """Evaluate the model on the MMLU benchmark."""
        # needs to instantiate self.results and self.responses
        # for save_results to function
        
        # get IDs for target tokens
        id_a = self.tokenizer.convert_tokens_to_ids(" A")
        id_b = self.tokenizer.convert_tokens_to_ids(" B")
        id_c = self.tokenizer.convert_tokens_to_ids(" C")
        id_d = self.tokenizer.convert_tokens_to_ids(" D")

        # infer device from model
        device = next(self.model.parameters()).device

        # evaluation loop
        batch_size = self.config["evaluation"]["batch_size"]

        with torch.no_grad():
            for i in range(0, len(self.data), batch_size):
                # create batch
                batch = self.data.select(range(i, min(i + batch_size, 
                                                      len(self.data))))
                # tokenize prompts and move to device
                tokenized_prompts = self.tokenizer(batch["prompt"], 
                                                   return_tensors="pt",
                                                   padding=True, 
                                                   truncation=True)
                tokenized_prompts = {K: v.to(device) for k, v in 
                                     tokenized_prompts.items()}

"""
with torch.no_grad():
  for weird slicing thing:
    tokenize prompt (returns input_ids, attention_mask)
    forward pass
    find last token in seq_length dimension with mask = 1 (across all ex
    in batch at once: seq_length - 1)
    record the prompt for all examples in batch
    for the last real token:
      grab the logits for the 4 token IDs we identified
      return argmax(token_ids)
      record reported LM answer
      compare to answer key and tally correct answer
"""