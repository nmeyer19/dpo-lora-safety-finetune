import yaml
from data.loaders.mmlu import MMLUDataLoader

with open("./configs/mmlu.yaml") as f:
    config = yaml.safe_load(f)

loader = MMLUDataLoader(config)
loader.load()
data = loader.get_data()
print(data[0]["answer"])
print(type(data[0]["answer"]))