# SemantiQ: Bloom's Taxonomy Question Classifier

Classify educational questions into Bloom's Taxonomy cognitive levels using **S-BERT** and **S-FastText** models with semantic enrichment.

## Overview

This project implements the methodology from the research paper *"Semantic-BERT and semantic-FastText models for education question classification"*. It classifies questions into 6 cognitive levels:

| Level | Description | Example |
|-------|-------------|---------|
| **Remember** | Recall facts and basic concepts | "What is the capital of France?" |
| **Understand** | Explain ideas or concepts | "Describe how photosynthesis works." |
| **Apply** | Use information in new situations | "Calculate the area of a triangle." |
| **Analyze** | Draw connections among ideas | "Compare and contrast mitosis and meiosis." |
| **Evaluate** | Justify a decision or course of action | "Assess the effectiveness of this policy." |
| **Create** | Produce new or original work | "Design an experiment to test plant growth." |

## Key Features

- **Semantic Dependency Parsing**: Uses spaCy to extract word relationships (QW, ROOT, NOUN, PROPN) for intent detection
- **Data Augmentation**: LLM-powered generation (Ollama) with semantic filtering to address class imbalance
- **Dual Model Architecture**: S-FastText (fast, lightweight) and S-BERT (higher accuracy)
- **Streamlit Web App**: Interactive classification with probability visualizations
- **5W1H Coverage**: Optimized for Who, What, Where, When, Why, How questions

## Project Structure

```
SemantiQ/
├── app.py                      # Streamlit Web Application
├── config/
│   └── config.yaml             # Hyperparameters
├── data/
│   ├── raw/                    # Original + augmented datasets
│   ├── processed/              # Train/val/test splits (CSV)
│   └── dataset/                # Source data files
├── docs/
│   ├── presentations/          # Project presentations (.pptx)
│   └── research paper (PDF)
├── models/
│   ├── s-fasttext.bin          # Trained FastText model
│   ├── s-bert.pth              # BERT state dictionary
│   └── s-bert_model/           # Full BERT model directory
├── notebooks/
│   ├── 01_eda.ipynb            # Exploratory data analysis
│   ├── 02_fasttext_train.ipynb # FastText training
│   ├── 03_bert_train.ipynb     # BERT training
│   ├── 04_inference_demo.ipynb # Model inference demo
│   ├── 05_data_augmentation.ipynb # LLM-based data augmentation
│   └── final.ipynb             # Complete training pipeline
├── scripts/                    # Utility scripts
├── src/
│   ├── preprocessing.py        # Text cleaning + Semantic Parsing
│   ├── dataset.py              # PyTorch datasets
│   ├── fasttext_classifier.py  # S-FastText implementation
│   ├── bert_classifier.py      # S-BERT implementation
│   ├── evaluate.py             # Metrics and evaluation
│   └── inference.py            # Unified prediction API
├── logs/                       # Training logs
└── requirements.txt
```

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### 2. Run Streamlit App

```bash
streamlit run app.py
```

### 3. Train Models (Optional)

Run the complete training pipeline:
```bash
# Open and run notebooks/final.ipynb
# Or run the individual training notebooks
```

### 4. Run Inference (CLI)

```python
from src.inference import BloomClassifier

classifier = BloomClassifier(model_type="fasttext")
result = classifier.predict("What is photosynthesis?")
# {'level': 'Remember', 'confidence': 0.92}

classifier = BloomClassifier(model_type="bert")
result = classifier.predict("Design an experiment to test plant growth")
# {'level': 'Create', 'confidence': 0.95}
```

## Model Configurations

### S-FastText (Algorithm 2)
| Parameter | Value |
|-----------|-------|
| Learning Rate | 0.3 |
| Epochs | 10 |
| Word N-grams | 2 |
| Embedding Dim | 100 |

### S-BERT (Algorithm 3)
| Parameter | Value |
|-----------|-------|
| Base Model | bert-base-uncased |
| Max Length | 128 |
| Batch Size | 16 |
| Learning Rate | 4e-5 |
| Epochs | 3 |

## Data Augmentation Pipeline

The `05_data_augmentation.ipynb` notebook implements:

1. **Distribution Analysis**: Identify minority classes needing augmentation
2. **Expert Prompting**: Use Ollama LLM with "Data Engineer" persona
3. **5W1H Constraints**: Enforce question diversity (How, Why, What, Imperatives)
4. **Semantic Filtering**: Validate questions using spaCy (ROOT verb check)
5. **Merge & Export**: Combine synthetic data with original training set

## Semantic Enrichment

Text preprocessing enriches questions with dependency tags:
```
Input:  "What is an atom?"
Output: "what_QW is_ROOT atom_NOUN"
```

Tags applied:
- `_QW`: Wh-words (What, Why, How, etc.)
- `_ROOT`: Main action verb
- `_NOUN` / `_PROPN`: Nouns and proper nouns

## Evaluation

Run evaluation on test set:

```bash
python src/evaluate.py --model both --test-data data/processed/test.csv
```

Outputs:
- Accuracy, Precision, Recall, F1-score
- Confusion matrices for each model
- Model comparison charts

## Research Reference

Based on the paper:
> **"Semantic-BERT and semantic-FastText models for education question classification"**

Key contributions:
- Semantic Dependency Parsing for intent detection
- Functional role enrichment of question text
- High accuracy on 5W1H educational questions

## Author

**ER-RAHOUTI ACHRAF**

- GitHub: [@achrafS133](https://github.com/achrafS133)

## License

MIT License
