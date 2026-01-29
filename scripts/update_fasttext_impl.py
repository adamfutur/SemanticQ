import json

nb_path = r'c:\Users\MSI\Desktop\SemanticQ\notebooks\semantic_bloom_full-update.ipynb'
with open(nb_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# 1. New FastText Training Source
ft_training_source = [
    "import fasttext\n",
    "\n",
    "print(\"Preparing FastText data files...\")\n",
    "# FastText supervised training requires a text file with __label__Lable format\n",
    "def prepare_fasttext_data(df, filename):\n",
    "    with open(filename, 'w', encoding='utf-8') as f:\n",
    "        for _, row in df.iterrows():\n",
    "            # Ensure label has no spaces (Bloom's levels usually don't, but good practice)\n",
    "            label = row['label'].replace(' ', '_') \n",
    "            text = row['clean_text']\n",
    "            f.write(f\"__label__{label} {text}\\n\")\n",
    "\n",
    "prepare_fasttext_data(train_df, 'bloom_train.txt')\n",
    "prepare_fasttext_data(val_df, 'bloom_val.txt')\n",
    "prepare_fasttext_data(test_df, 'bloom_test.txt')\n",
    "\n",
    "print(\"Training S-FastText model (Supervised implementation as per Paper Algorithm 2)...\")\n",
    "# Parameters from Paper Table 2: \n",
    "# Learning Rate = 0.3, Epoch = 10, Word N-grams = 2\n",
    "ft_model = fasttext.train_supervised(\n",
    "    input='bloom_train.txt',\n",
    "    lr=0.3, \n",
    "    epoch=10, \n",
    "    wordNgrams=2, \n",
    "    verbose=2\n",
    ")\n",
    "\n",
    "# Evaluation on Validation Set (Validation split from Training Phase)\n",
    "print(\"\\nFastText Validation Evaluation:\")\n",
    "val_texts = val_df['clean_text'].tolist()\n",
    "val_labels = val_df['label'].tolist()\n",
    "\n",
    "# Predict\n",
    "val_preds_raw = ft_model.predict(val_texts)\n",
    "val_preds = [label[0].replace('__label__', '') for label in val_preds_raw[0]]\n",
    "\n",
    "print(classification_report(val_labels, val_preds))\n",
    "ft_val_acc = accuracy_score(val_labels, val_preds)\n",
    "print(f\"FastText Validation Accuracy: {ft_val_acc:.4f}\")"
]

# 2. Updated Evaluation Cell Source (Cell 11 approx)
eval_source = [
    "def evaluate_model(y_true, y_pred, name):\n",
    "    print(f\"\\nEvaluation results for {name}:\")\n",
    "    print(classification_report(y_true, y_pred, target_names=config['labels']))\n",
    "    \n",
    "    cm = confusion_matrix(y_true, y_pred)\n",
    "    plt.figure(figsize=(10, 8))\n",
    "    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', \n",
    "                xticklabels=config['labels'], yticklabels=config['labels'])\n",
    "    plt.title(f\"Confusion Matrix - {name}\")\n",
    "    plt.ylabel('True Label')\n",
    "    plt.xlabel('Predicted Label')\n",
    "    plt.show()\n",
    "\n",
    "# FastText Predictions (Supervised)\n",
    "print(\"Predicting with S-FastText on Test Set...\")\n",
    "test_texts = test_df['clean_text'].tolist()\n",
    "ft_raw = ft_model.predict(test_texts)\n",
    "ft_pred_labels = [p[0].replace('__label__', '') for p in ft_raw[0]]\n",
    "\n",
    "# Convert string labels to IDs to match y_test\n",
    "# Ensure label_to_id exists. If not, recreate from id_to_label\n",
    "if 'label_to_id' not in globals():\n",
    "    label_to_id = {v: k for k, v in id_to_label.items()}\n",
    "\n",
    "ft_preds = [label_to_id[l] for l in ft_pred_labels]\n",
    "\n",
    "evaluate_model(y_test, ft_preds, \"Semantic-FastText\")\n",
    "\n",
    "# BERT Predictions\n",
    "test_ds = BertDataset(test_df['clean_text'].tolist(), y_test, tokenizer, config['bert']['max_length'])\n",
    "test_loader = DataLoader(test_ds, batch_size=config['bert']['batch_size'])\n",
    "\n",
    "bert_model.eval()\n",
    "bert_preds = []\n",
    "with torch.no_grad():\n",
    "    for batch in tqdm(test_loader, desc=\"BERT Inference\"):\n",
    "        input_ids = batch['input_ids'].to(device)\n",
    "        attention_mask = batch['attention_mask'].to(device)\n",
    "        outputs = bert_model(input_ids, attention_mask=attention_mask)\n",
    "        preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()\n",
    "        bert_preds.extend(preds)\n",
    "\n",
    "evaluate_model(y_test, np.array(bert_preds), \"Semantic-BERT\")"
]

# 3. Updated Inference Cell Source (Cell 15 approx)
inference_source = [
    "def predict_question(question):\n",
    "    clean_q = preprocessor.clean_text(question)\n",
    "    \n",
    "    # FastText Prediction (Supervised)\n",
    "    # predict returns (['__label__Label'], [prob])\n",
    "    ft_res = ft_model.predict(clean_q)\n",
    "    ft_label_str = ft_res[0][0].replace('__label__', '')\n",
    "    ft_prob = ft_res[1][0]\n",
    "    \n",
    "    # BERT Prediction\n",
    "    inputs = tokenizer(clean_q, return_tensors='pt', truncation=True, padding='max_length', max_length=128).to(device)\n",
    "    with torch.no_grad():\n",
    "        outputs = bert_model(**inputs)\n",
    "        bert_idx = torch.argmax(outputs.logits, dim=1).item()\n",
    "        bert_prob = torch.softmax(outputs.logits, dim=1).max().item()\n",
    "    bert_label = id_to_label[bert_idx]\n",
    "    \n",
    "    print(f\"\\nQuestion: \\\"{question}\\\"\")\n",
    "    print(f\"  → FastText: {ft_label_str} ({ft_prob:.2%})\")\n",
    "    print(f\"  → BERT:     {bert_label} ({bert_prob:.2%})\")\n",
    "\n",
    "# Sample tests\n",
    "predict_question(\"Can you describe the components of an atom?\")\n",
    "predict_question(\"How would you categorize the main themes of the story?\")\n",
    "predict_question(\"Create a hypothetical scenario where the law of gravity is reversed.\")"
]

files_updated = 0
for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = "".join(cell['source'])
        if "Training FastText embeddings..." in source:
            cell['source'] = ft_training_source
            files_updated += 1
            print("Updated FastText Training Cell")
        elif "evaluate_model(y_test, ft_preds, \"Semantic-FastText\")" in source:
             cell['source'] = eval_source
             files_updated += 1
             print("Updated Evaluation Cell")
        elif "predict_question(\"Can you describe the components of an atom?\")" in source:
             cell['source'] = inference_source
             files_updated += 1
             print("Updated Inference Cell")

if files_updated > 0:
    with open(nb_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1)
    print(f"Notebook saved with {files_updated} cell updates.")
else:
    print("No cells matched.")
