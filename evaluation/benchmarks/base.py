from abc import ABC, abstractmethod
from transformers import PreTrainedTokenizer, PreTrainedModel
import os, json, csv

class BaseBenchmark(ABC):
    """
    Abstract base class for both evaluation benchmarks. 
    
    All benchmarks should inherit from this class and must implement the
    `load` method to load the relevant benchmark and the `evaluate` method, 
    which should run the eval and populate `self.results` and `self.responses`.
    Every benchmark receives a config dictionary, a model being evaluated,
    and its tokenizer.
    The run() method will run the full eval pipeline and save the results in
    the location per the config passed in at initialization. 
    """
    def __init__(self, config: dict, model: PreTrainedModel, 
        tokenizer: PreTrainedTokenizer
    ) -> None:
        """Initialize the benchmark with the provided config, model, and tokenizer."""
        self.config = config
        self.model = model
        self.tokenizer = tokenizer

    def run(self) -> None:
        """Load, evaluate, and save results."""
        self.load()
        self.evaluate()
        self.save_results()

    @abstractmethod
    def load(self) -> None:
        """Load and preprocess the benchmark."""
        ...

    @abstractmethod
    def evaluate(self) -> None:
        """Evaluate the model on the benchmark."""
        # needs to instantiate self.results and self.responses
        # for save_results to function
        ...

    def save_results(self) -> None:
        """Saves results of the model evaluation."""
        # make directory if it doesn't exist
        path = self.config["outputs"]["results_dir"]
        os.makedirs(path, exist_ok=True)
        
        # save high-level info in a json file
        with open(path + "/results.json", "w") as file:
            json.dump(self.results, file)
        
        # save all outputs in a csv file
        with open(path + "/results.csv", "w") as file:
            writer = csv.DictWriter(file, fieldnames=self.responses[0].keys())
            writer.writeheader()
            writer.writerows(self.responses)