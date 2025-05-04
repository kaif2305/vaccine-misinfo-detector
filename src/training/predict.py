"""
predict(text: str) -> label

Example Usage:
detector = VaccineMisinformationDetector(model_path="saved_model/")
label = detector.predict("Vaccines cause autism.")

@Author: Xinrui WAN
@Date: 29/Apr/2025
"""
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, BertTokenizer
import logging
import re
import os
import pandas as pd
from sklearn.model_selection import train_test_split

class VaccineMisinformationDetector:
    def __init__(self, model_path: str):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.max_len = 128
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

    def preprocess_for_inference(self, text) -> tuple:
        text = text.lower().strip()
        encoding = self.tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=self.max_len,
            return_tensors='pt'
        )
        return encoding['input_ids'].to(self.device), encoding['attention_mask'].to(self.device)

    def predict(self, text: str) -> int:
        logging.debug(f"Predicting label for text: {text}")
        cleaned_text = self.clean_text(text)
        preprocess_and_tokenized_text = self.preprocess_for_inference(cleaned_text)

        with torch.no_grad():
            outputs = self.model(preprocess_and_tokenized_text[0], attention_mask=preprocess_and_tokenized_text[1])
            logits = outputs.logits
            probs = torch.softmax(logits, dim=1)
            pred_label = torch.argmax(probs, dim=1).item()

        return pred_label

#TODO Seems model is not well trained, all the input is classified as 0
if __name__ == '__main__':
    # Example usage
    detector = VaccineMisinformationDetector(model_path="heishi99/vaccine-misinfo-bert")
    claim = "Vaccines cause autism."
    label = detector.predict(claim)
    print(f"Claim: {claim}\nPredicted label: {label}")
    claim = "COVID-19 vaccines do not affect fertility."
    label = detector.predict(claim)
    print(f"Claim: {claim}\nPredicted label: {label}")