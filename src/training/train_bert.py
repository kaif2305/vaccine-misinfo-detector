import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from transformers import BertForSequenceClassification, get_scheduler
from torch.optim import AdamW
from tqdm import tqdm
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_dataset(split="train"):
    input_ids = torch.load(f"../../data/processed/{split}_input_ids.pt")
    attention_masks = torch.load(f"../../data/processed/{split}_attention_mask.pt")
    labels = torch.load(f"../../data/processed/{split}_labels.pt")

    dataset = TensorDataset(input_ids, attention_masks, labels)
    return dataset

train_dataset = load_dataset("train")
val_dataset = load_dataset("val")
test_dataset = load_dataset("test") 

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
model.to(device)


optimizer = AdamW(model.parameters(), lr=2e-5)
num_epochs = 4
total_steps = len(train_loader) * num_epochs

lr_scheduler = get_scheduler(
    "linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=total_steps
)


criterion = nn.CrossEntropyLoss()


for epoch in range(num_epochs):
    model.train()
    total_loss = 0

    for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}"):
        print(f"Batch size: {len(batch)}")
        print(f"Batch data: {batch}")
        b_input_ids = batch[0].to(device)
        b_attention_mask = batch[1].to(device)
        b_labels = batch[2].to(device)

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

    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch + 1} | Training Loss: {avg_loss:.4f}")


    model.eval()
    val_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in val_loader:
            b_input_ids = batch[0].to(device)
            b_attention_mask = batch[1].to(device)
            b_labels = batch[2].to(device)

            outputs = model(
                input_ids=b_input_ids,
                attention_mask=b_attention_mask
            )
            logits = outputs.logits
            preds = torch.argmax(logits, dim=1)

            correct += (preds == b_labels).sum().item()
            total += b_labels.size(0)

    val_acc = correct / total
    print(f"Epoch {epoch + 1} | Validation Accuracy: {val_acc:.4f}")


os.makedirs("models/bert_saved", exist_ok=True)
model.save_pretrained("models/bert_saved")
