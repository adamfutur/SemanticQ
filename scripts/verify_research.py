import pandas as pd
import numpy as np
import spacy
import fasttext
import os
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

print("🚀 Starting Semantic-FastText Research Implementation...")

# Load spaCy
nlp = spacy.load("en_core_web_md")

def assign_qw_tag(token):
    interrogatives = {'who', 'what', 'where', 'when', 'why', 'how', 'which', 'whom'}
    if token.text.lower() in interrogatives:
        return "QW"
    return token.tag_

def semantic_parse(text):
    doc = nlp(text)
    enriched_tokens = []
    roots = [token for token in doc if token.head == token]
    root = roots[0] if roots else doc[0]
    
    for token in doc:
        if token.is_punct or token.is_space:
            continue
        qw_tag = assign_qw_tag(token)
        dep = token.dep_
        feature = f"{token.lemma_.lower()}"
        if qw_tag == "QW": feature = f"QW_{feature}"
        feature = f"{feature}_{dep}"
        if token == root: feature = f"ROOT_{feature}"
        enriched_tokens.append(feature)
        
    return " ".join(enriched_tokens)

# Load Data
current_dir = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(os.path.join(current_dir, 'data', 'raw', 'bloom_questions.csv'))
print(f"📦 Dataset: {len(df)} samples")

print("🔍 Applying Semantic Dependency Parsing...")
df['clean_text'] = df['question'].apply(semantic_parse)

def evaluate():
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = []
    
    print("\n--- 5-Fold Stratified Cross-Validation ---")
    
    for fold, (train_idx, test_idx) in enumerate(skf.split(df['clean_text'], df['level'])):
        train_df, test_df = df.iloc[train_idx], df.iloc[test_idx]
        
        # Prepare data
        with open('temp_train.txt', 'w', encoding='utf-8') as f:
            for _, row in train_df.iterrows():
                f.write(f"__label__{row['level']} {row['clean_text']}\n")
        
        # Train (Specs from Paper)
        model = fasttext.train_supervised(
            input='temp_train.txt',
            lr=0.3,
            epoch=10,
            wordNgrams=2
        )
        
        # Predict
        labels, _ = model.predict(test_df['clean_text'].tolist())
        preds = [l[0].replace('__label__', '') for l in labels]
        
        # Metrics
        acc = accuracy_score(test_df['level'], preds)
        p, r, f1, _ = precision_recall_fscore_support(test_df['level'], preds, average='weighted')
        
        print(f"Fold {fold+1}: Accuracy = {acc:.4f}, F1 = {f1:.4f}")
        scores.append(acc)
        
    print(f"\n✅ Final Average Accuracy: {np.mean(scores):.4f}")

if __name__ == "__main__":
    evaluate()
