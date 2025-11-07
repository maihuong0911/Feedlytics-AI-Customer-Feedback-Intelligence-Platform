from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Load PhoBERT pre-trained (sentiment classification, đã fine-tune sẵn)
model_name = "vinai/phobert-base"

print("Loading PhoBERT...")
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)

# Ví dụ 1 câu tiếng Việt
sentence = "Sản phẩm này rất tuyệt vời!"
inputs = tokenizer(sentence, return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs)

logits = outputs.logits
pred = torch.argmax(logits, dim=1).item()

print("Sentence:", sentence)
print("Logits:", logits)
print("Predicted label:", pred)
