"""
Data clean module for CoAID dataset, only for news and claims, not with tweets.

@Author: Xinrui WAN
@Date: 20/Apr/2025
"""
import pandas as pd
import os
import re
from sklearn.model_selection import train_test_split

# ========== Step 1: Text Cleaning Function ==========
def clean_text(text):
    text = re.sub(r"http\S+|www\S+|https\S+", '', text)  # remove URLs
    text = re.sub(r"@\w+", '', text)  # remove @mentions
    text = re.sub(r"#", '', text)  # remove hashtag symbol
    # text = re.sub(r"[^\w\s.,!?]", '', text)  # remove special characters except punctuation
    text = re.sub(r"\s+", ' ', text).strip()  # remove extra spaces
    return text

# ========== Step 2: Data Loading ==========
path_dataset = ["raw/CoAID-0.4"]
path_folders = ["05-01-2020", "07-01-2020", "09-01-2020", "11-01-2020"]
path_files = ["ClaimFakeCOVID-19.csv", "ClaimRealCOVID-19.csv", "NewsFakeCOVID-19.csv", "NewsRealCOVID-19.csv"]

all_data = []

for dataset in path_dataset:
    for folder in path_folders:
        folder_path = os.path.join(dataset, folder)
        for file in path_files:
            file_path = os.path.join(folder_path, file)
            if os.path.exists(file_path):
                df = pd.read_csv(file_path)

                if 'ClaimFake' in file:
                    texts = df['title'].dropna().tolist()
                    labels = [1] * len(texts)
                elif 'ClaimReal' in file:
                    texts = df['title'].dropna().tolist()
                    labels = [0] * len(texts)
                elif 'NewsFake' in file:
                    texts = []
                    for idx, row in df.iterrows():
                        if pd.notna(row.get('content')) and len(row['content']) > 50:
                            texts.append(row['content'])
                        elif pd.notna(row.get('title')):
                            texts.append(row['title'])
                    labels = [1] * len(texts)
                elif 'NewsReal' in file:
                    texts = []
                    for idx, row in df.iterrows():
                        if pd.notna(row.get('content')) and len(row['content']) > 50:
                            texts.append(row['content'])
                        elif pd.notna(row.get('title')):
                            texts.append(row['title'])
                    labels = [0] * len(texts)

                cleaned_texts = [clean_text(t) for t in texts]

                for text, label in zip(cleaned_texts, labels):
                    all_data.append({'text': text, 'label': label})

# Convert to DataFrame
final_df = pd.DataFrame(all_data)

# Remove duplicates
final_df = final_df.drop_duplicates(subset=['text'])

# Filter out too short texts (<10 words)
final_df = final_df[final_df['text'].apply(lambda x: len(x.split()) >= 10)]

# ========== Step 3: Balance the Classes ==========
# Separate classes
factual = final_df[final_df['label'] == 0]
misinfo = final_df[final_df['label'] == 1]

# Downsample the majority class
min_size = min(len(factual), len(misinfo))
factual_balanced = factual.sample(n=min_size, random_state=42)
misinfo_balanced = misinfo.sample(n=min_size, random_state=42)

# Combine and shuffle
balanced_df = pd.concat([factual_balanced, misinfo_balanced]).sample(frac=1, random_state=42).reset_index(drop=True)

# ========== Step 4: Split into Train/Val/Test ==========
train_val, test = train_test_split(balanced_df, test_size=0.1, stratify=balanced_df['label'], random_state=42)
train, val = train_test_split(train_val, test_size=0.111, stratify=train_val['label'], random_state=42)  # 0.111 * 0.9 = 0.1

# ========== Step 5: Save Files ==========
os.makedirs("processed", exist_ok=True)
train.to_csv("processed/train.csv", index=False, header=["text", "label"])
val.to_csv("processed/val.csv", index=False, header=["text", "label"])
test.to_csv("processed/test.csv", index=False, header=["text", "label"],)
print("Processing complete! Saved train.csv, val.csv, and test.csv.")