from datasets import load_dataset
from data.loaders.base import BaseDataLoader

class DollyDataLoader(BaseDataLoader):
    """
    Data loader for the Databricks dolly-15k dataset.
    Loads a subset of samples from the dataset, concatenates the instruction
    and context fields into a single prompt, and formats everything into 
    (prompt, response) tuples for SFT.
    """

    def load(self) -> None:
        """Load and preprocess the dolly-15k data."""
        cfg = self.config["data"]
        raw_dataset = load_dataset(cfg["dataset"], split=cfg["split"])

        if cfg["max_samples"] is not None:
            raw_dataset = raw_dataset.shuffle(seed=cfg["seed"]).select(range(cfg["max_samples"]))
        
        self.data = raw_dataset.map(self._format_pair)
    
    def _format_pair(self, example: dict) -> dict:
        """
        Formats a raw dolly example into a (prompt, response) pair for SFT.
        each dolly-15k example contains instruction, context, response, and
        category fields. We concatenate the instruction and context fields 
        and return the desired tuple structure.
        """

        instruction = example["instruction"]
        context = example["context"]
        response = example["response"]

        if context:
            prompt = f"Instruction: {instruction}\nContext: {context}\nResponse:"
        else: 
            prompt = f"Instruction: {instruction}\nResponse:"
        
        return {
            "prompt": prompt,
            "response": response
        }