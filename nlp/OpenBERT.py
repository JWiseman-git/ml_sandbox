from transformers import AutoTokenizer, AutoModelForMaskedLM

# It is best practice to load the tokenizer and model together
model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForMaskedLM.from_pretrained(model_name)

print(model)