from datasets import load_dataset
from data.loaders.base import BaseDataLoader

class AdvBenchDataLoader(BaseDataLoader):
    """
    Data loader for the AdvBench dataset.
    Loads the set of prompts from AdvBench.
    """

    def load(self) -> None:
        """Load and preprocess the AdvBench dataset."""
        cfg = self.config["benchmark"]
        raw_dataset = load_dataset(cfg["dataset"], split=cfg["split"])

        self.data = raw_dataset.select_columns([cfg["prompt_column"]])