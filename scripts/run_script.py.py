#!/usr/bin/env python
# coding: utf-8

# # Anti-Graphiti: Semantic-BERT & Semantic-FastText Implementation
# 
# This notebook provides the implementation of the project based on the research paper **"Semantic-BERT and semantic-FastText models for education question classification"**.
# 
# ## Objectives:
# 1. Implementation of Semantic Dependency Parsing using spaCy.
# 2. S-FastText with sub-word n-gram embeddings.
# 3. S-BERT with BERT-Large architecture (24 blocks, 16 heads, 1024 hidden size).
# 4. 5-Fold Stratified Cross-Validation evaluation.

# ### # Cell 1: Environment Setup
# This cell installs all necessary libraries and downloads the required language model for spaCy.

# In[ ]:


get_ipython().system('pip install spacy transformers fasttext scikit-learn pandas tqdm torch matplotlib seaborn openpyxl')
get_ipython().system('python -m spacy download en_core_web_md')

import pandas as pd
import numpy as np
import spacy
import torch
import fasttext
import os
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

nlp = spacy.load("en_core_web_md")
print("Environment initialized.")


# ### # Cell 2: Preprocessing & Semantic Parsing
# Implementation of the Semantic Dependency Parsing algorithm as described in the research paper.

# In[ ]:


def assign_qw_tag(token):
    """Assign custom 'QW' tag to interrogative words."""
    interrogatives = {'who', 'what', 'where', 'when', 'why', 'how', 'which', 'whom'}
    if token.text.lower() in interrogatives:
        return "QW"
    return token.tag_

def semantic_parse(text):
    """
    Implements Semantic Dependency Parsing:
    - Identifies Root word and relationships
    - Assigns QW tags
    - Enriches tokens with functional roles
    """
    doc = nlp(text)
    enriched_tokens = []

    # Identify root word (usually the main verb/action)
    roots = [token for token in doc if token.head == token]
    root = roots[0] if roots else doc[0]

    for token in doc:
        if token.is_punct or token.is_space:
            continue

        qw_tag = assign_qw_tag(token)
        dep = token.dep_

        feature = f"{token.lemma_.lower()}"
        if qw_tag == "QW":
            feature = f"QW_{feature}"

        feature = f"{feature}_{dep}"

        if token == root:
            feature = f"ROOT_{feature}"

        enriched_tokens.append(feature)

    return " ".join(enriched_tokens)

# Load Dataset (adjust path if necessary)
DATA_PATH = r'../data/raw/bloom_questions.csv'
df = pd.read_csv(DATA_PATH)
df['clean_text'] = df['question'].apply(semantic_parse)
print(f"Loaded {len(df)} samples with semantic enrichment.")
df[['question', 'level', 'clean_text']].head()


# ### # Cell 3: Model 1 - S-FastText Implementation
# Builds and evaluates the S-FastText model with Word N-grams = 2 and Learning Rate = 0.3.

# In[ ]:


def train_s_fasttext(train_df, val_df):
    """Build S-FastText using parameters from paper specifications."""
    def to_ft_format(df, filename):
        with open(filename, 'w', encoding='utf-8') as f:
            for _, row in df.iterrows():
                f.write(f"__label__{row['level']} {row['clean_text']}\n")

    to_ft_format(train_df, 'ft_train.txt')

    model = fasttext.train_supervised(
        input='ft_train.txt',
        lr=0.3,
        epoch=10,
        wordNgrams=2,
        dim=100,
        loss='softmax'
    )

    labels, probs = model.predict(val_df['clean_text'].tolist())
    preds = [l[0].replace('__label__', '') for l in labels]

    return preds, val_df['level'].tolist()

print("S-FastText implementation defined.")


# ### # Cell 4: Model 2 - S-BERT Implementation
# Implements BERT-Large with specified hyperparameters and data processing.

# In[ ]:


class QuestionDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
        unique_labels = sorted(list(set(labels)))
        self.label_map = {l: i for i, l in enumerate(unique_labels)}

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        text = str(self.texts[item])
        label = self.label_map[self.labels[item]]
        encoding = self.tokenizer.encode_plus(
            text, add_special_tokens=True, max_length=self.max_len,
            padding='max_length', truncation=True, return_attention_mask=True,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

def train_s_bert(train_df, val_df):
    """S-BERT implementation using BERT-Large architecture."""
    MODEL_NAME = 'bert-large-uncased' # Spec: 24 blocks, 1024 hidden size
    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_ds = QuestionDataset(train_df['clean_text'].tolist(), train_df['level'].tolist(), tokenizer)
    val_ds = QuestionDataset(val_df['clean_text'].tolist(), val_df['level'].tolist(), tokenizer)

    train_loader = DataLoader(train_ds, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=4)

    model = BertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=6)
    model.to(device)

    optimizer = AdamW(model.parameters(), lr=4e-5) # Spec: 0.00004

    # Training loop (e.g., 10 epochs as per request)
    model.train()
    for epoch in range(10):
        train_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()

    # Evaluation Phase
    model.eval()
    all_preds = []
    inv_label_map = {i: l for l, i in train_ds.label_map.items()}
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            outputs = model(input_ids, attention_mask=attention_mask)
            preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()
            all_preds.extend([inv_label_map[p] for p in preds])

    return all_preds, val_df['level'].tolist()


# ### # Cell 5: Evaluation & Validation
# Performs 5-Fold Stratified Cross-Validation and calculates aggregate metrics.

# In[ ]:


def evaluate_experiment(df):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = []

    for fold, (train_idx, test_idx) in enumerate(skf.split(df['clean_text'], df['level'])):
        print(f"Executing Fold {fold+1}...")
        train_df, test_df = df.iloc[train_idx], df.iloc[test_idx]

        y_pred, y_true = train_s_fasttext(train_df, test_df)

        acc = accuracy_score(y_true, y_pred)
        p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted')
        scores.append({'fold': fold+1, 'accuracy': acc, 'precision': p, 'recall': r, 'f1': f1})

        if fold == 0:
            cm = confusion_matrix(y_true, y_pred)
            sns.heatmap(cm, annot=True, fmt='d', xticklabels=sorted(list(set(df['level']))))
            plt.title(f"Confusion Matrix - Fold {fold+1}")
            plt.show()

    results = pd.DataFrame(scores)
    print("\n--- Overall Cross-Validation Results ---")
    print(results.mean())

evaluate_experiment(df)

