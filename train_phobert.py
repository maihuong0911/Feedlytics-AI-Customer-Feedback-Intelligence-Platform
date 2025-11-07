import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from datasets import Dataset
import torch
import numpy as np

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"⚙️ Đang sử dụng thiết bị: {device}")

# Load data
print("🚀 Đang đọc dữ liệu ...")
df = pd.read_csv("train.csv")
df = df.dropna(subset=["text"])
df["text"] = df["text"].astype(str)

# Encode labels
print("🔢 Mã hóa nhãn ...")
label_map = {"negative": 0, "neutral": 1, "positive": 2}
df["label"] = df["label"].map(label_map)
print(f"✅ Các nhãn: {list(label_map.keys())}")

# Convert to Dataset
dataset = Dataset.from_pandas(df)
train_test_split = dataset.train_test_split(test_size=0.2)
train_dataset = train_test_split["train"]
test_dataset = train_test_split["test"]

# Load tokenizer
print("🔤 Đang tải tokenizer ...")
model_name = "vinai/phobert-base"
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)

# Tokenize function
def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        padding="max_length",
        truncation=True,
        max_length=128
    )

# Tokenize datasets
print("📦 Tokenizing dữ liệu ...")
train_dataset = train_dataset.map(tokenize_function, batched=True)
test_dataset = test_dataset.map(tokenize_function, batched=True)

# Load model
print("🧠 Đang tải mô hình PhoBERT ...")
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=3
).to(device)

# Training arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    logging_dir="./logs",
    logging_steps=10,
    save_strategy="epoch",
    load_best_model_at_end=True,
)

# Initialize Trainer
print("🔥 Bắt đầu huấn luyện ...")
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
)

# Train the model
trainer.train()

# Save the model
trainer.save_model("./phobert_finetuned")
tokenizer.save_pretrained("./phobert_finetuned")
print("🎉 Mô hình đã được lưu!")