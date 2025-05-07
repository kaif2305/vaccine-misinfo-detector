import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from transformers import BertForSequenceClassification, get_scheduler
from torch.optim import AdamW
from tqdm import tqdm
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_dataset(split="train"):
    input_ids = torch.load(f"../data/processed/{split}_input_ids.pt")
    attention_masks = torch.load(f"../data/processed/{split}_attention_mask.pt")
    labels = torch.load(f"../data/processed/{split}_labels.pt")
    return TensorDataset(input_ids, attention_masks, labels)

train_loader = DataLoader(load_dataset("train"), batch_size=16, shuffle=True)
val_loader = DataLoader(load_dataset("val"), batch_size=16, shuffle=False)
test_loader = DataLoader(load_dataset("test"), batch_size=16, shuffle=False)

model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
model.to(device)

optimizer = AdamW(model.parameters(), lr=2e-5)
num_epochs = 6
total_steps = len(train_loader) * num_epochs
lr_scheduler = get_scheduler(
    "linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=total_steps
)

criterion = nn.CrossEntropyLoss()
best_val_acc = 0.0
best_model_state = None

for epoch in range(num_epochs):
    model.train()
    total_loss = 0.0

    for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}"):
        b_input_ids, b_attention_mask, b_labels = [x.to(device) for x in batch]

        outputs = model(
            input_ids=b_input_ids,
            attention_mask=b_attention_mask,
            labels=b_labels
        )

        loss = outputs.loss
        total_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        lr_scheduler.step()

    avg_train_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch + 1} | Avg Training Loss: {avg_train_loss:.4f}")

    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in val_loader:
            b_input_ids, b_attention_mask, b_labels = [x.to(device) for x in batch]

            outputs = model(
                input_ids=b_input_ids,
                attention_mask=b_attention_mask
            )
            logits = outputs.logits
            preds = torch.argmax(logits, dim=1)

            val_loss += criterion(logits, b_labels).item()
            correct += (preds == b_labels).sum().item()
            total += b_labels.size(0)

    avg_val_loss = val_loss / len(val_loader)
    val_acc = correct / total
    print(f"Epoch {epoch + 1} | Val Accuracy: {val_acc:.4f} | Val Loss: {avg_val_loss:.4f}")

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_model_state = model.state_dict()
        print("New best model saved.")

if best_model_state:
    model.load_state_dict(best_model_state)

os.makedirs("../models/bert_saved", exist_ok=True)
model.save_pretrained("../models/bert_saved")
print("Training complete.")
