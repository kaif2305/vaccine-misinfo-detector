"""
predict(text: str) -> label

Example Usage:
detector = VaccineMisinformationDetector(model_path="saved_model/")
label = detector.predict("Vaccines cause autism.")

@Author: Xinrui WAN
@Date: 29/Apr/2025
"""
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import logging
import re
import os
import pandas as pd
from sklearn.model_selection import train_test_split

class VaccineMisinformationDetector:
    def __init__(self, model_path: str):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path) # Load the model
        self.model.to(self.device)
        self.model.eval()

    def clean_text(self, text: str) -> str:
        text = re.sub(r"http\S+|www\S+|https\S+", '', text)
        text = re.sub(r"@\w+", '', text)
        text = re.sub(r"#", '', text)
        #text = re.sub(r"[^\w\s.,!?]", '', text)
        text = re.sub(r"\s+", ' ', text).strip()
        return text

    def predict(self, text: str) -> int:
        logging.debug(f"Predicting label for text: {text}")
        cleaned_text = self.clean_text(text)
        inputs = self.tokenizer(cleaned_text, return_tensors='pt', truncation=True, padding=True, max_length=512)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=1)
            pred_label = torch.argmax(probs, dim=1).item()

        return pred_label

