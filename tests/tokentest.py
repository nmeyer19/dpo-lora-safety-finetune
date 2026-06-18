from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
id_a = tokenizer.encode(" A", add_special_tokens=False)[-1]
print(tokenizer.encode(" A", add_special_tokens=False))

print(tokenizer.decode([id_a]))

ids = tokenizer.encode("Answer: A", add_special_tokens=False)
print([tokenizer.decode([i]) for i in ids])