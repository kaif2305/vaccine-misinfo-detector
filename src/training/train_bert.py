import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from transformers import (
    BertForSequenceClassification,
    BertTokenizer,
    get_scheduler,
)
from torch.optim import AdamW
from tqdm import tqdm
import os
from huggingface_hub import HfApi, HfFolder

# Set up
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_NAME = "bert-base-uncased"
REPO_NAME = "vaccine-misinfo-bert"

tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
model = BertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
model.to(device)

def load_dataset(split="train"):
    input_ids = torch.load(f"data/processed/{split}_input_ids.pt")
    attention_masks = torch.load(f"data/processed/{split}_attention_mask.pt")
    labels = torch.load(f"data/processed/{split}_labels.pt")
    return TensorDataset(input_ids, attention_masks, labels)

train_loader = DataLoader(load_dataset("train"), batch_size=16, shuffle=True)
val_loader = DataLoader(load_dataset("val"), batch_size=16, shuffle=False)

optimizer = AdamW(model.parameters(), lr=2e-5)
num_epochs = 4
total_steps = len(train_loader) * num_epochs
lr_scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=total_steps)
criterion = nn.CrossEntropyLoss()

for epoch in range(num_epochs):
    model.train()
    total_loss = 0

    for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
        b_input_ids, b_attention_mask, b_labels = [t.to(device) for t in batch]

        outputs = model(input_ids=b_input_ids, attention_mask=b_attention_mask, labels=b_labels)
        loss = outputs.loss

        total_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        lr_scheduler.step()

    print(f"Epoch {epoch+1} | Training Loss: {total_loss/len(train_loader):.4f}")

    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in val_loader:
            b_input_ids, b_attention_mask, b_labels = [t.to(device) for t in batch]
            logits = model(input_ids=b_input_ids, attention_mask=b_attention_mask).logits
            preds = torch.argmax(logits, dim=1)
            correct += (preds == b_labels).sum().item()
            total += b_labels.size(0)
    val_acc = correct / total
    print(f"Epoch {epoch+1} | Validation Accuracy: {val_acc:.4f}")

save_dir = "models/bert_saved"
model.save_pretrained(save_dir, safe_serialization=True)
tokenizer.save_pretrained(save_dir)

from huggingface_hub import create_repo, push_to_hub

create_repo(REPO_NAME, exist_ok=True)
model.push_to_hub(REPO_NAME, safe_serialization=True)
tokenizer.push_to_hub(REPO_NAME)
from huggingface_hub import create_repo
from transformers import BertForSequenceClassification, BertTokenizer

create_repo("vaccine-misinfo-bert", private=True, exist_ok=True)


model.push_to_hub("vaccine-misinfo-bert")
tokenizer.push_to_hub("vaccine-misinfo-bert")