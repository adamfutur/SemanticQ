"""
Text Preprocessing Pipeline for Bloom's Taxonomy Question Classifier

This module provides utilities for cleaning, tokenizing, and preparing
educational questions for classification.
"""

import re
import string
from typing import List, Optional, Tuple

import nltk
import pandas as pd
import yaml
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split

# Download required NLTK data
def download_nltk_resources():
    """Download required NLTK resources."""
    resources = ['punkt', 'punkt_tab', 'stopwords', 'wordnet', 'omw-1.4']
    for resource in resources:
        try:
            nltk.download(resource, quiet=True)
        except Exception as e:
            print(f"Warning: Could not download {resource}: {e}")

download_nltk_resources()


def load_config(config_path: str = "config/config.yaml") -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


class TextPreprocessor:
    """
    Text preprocessing pipeline for educational questions.
    
    Attributes:
        lowercase: Convert text to lowercase
        remove_punctuation: Remove punctuation marks
        remove_stopwords: Remove common stopwords
        lemmatize: Apply lemmatization
        min_word_length: Minimum word length to keep
    """
    
    def __init__(
        self,
        lowercase: bool = True,
        remove_punctuation: bool = True,
        remove_stopwords: bool = False,
        lemmatize: bool = True,
        min_word_length: int = 2,
        use_semantic_parsing: bool = False
    ):
        self.lowercase = lowercase
        self.remove_punctuation = remove_punctuation
        self.remove_stopwords = remove_stopwords
        self.lemmatize = lemmatize
        self.min_word_length = min_word_length
        self.use_semantic_parsing = use_semantic_parsing
        
        self.lemmatizer = WordNetLemmatizer() if lemmatize else None
        self.stop_words = set(stopwords.words('english')) if remove_stopwords else set()
        
        self.nlp = None
        if use_semantic_parsing:
            import spacy
            try:
                self.nlp = spacy.load("en_core_web_md")
            except Exception:
                # Fallback if md model is not found
                try:
                    self.nlp = spacy.load("en_core_web_sm")
                except Exception:
                    print("Warning: SpaCy models not found. Semantic parsing disabled.")
                    self.use_semantic_parsing = False
    
    def clean_text(self, text: str) -> str:
        """
        Clean and preprocess a single text string.
        
        Args:
            text: Input text to preprocess
            
        Returns:
            Cleaned and preprocessed text
        """
        if not isinstance(text, str):
            return ""
        
        # Lowercase
        if self.lowercase:
            text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\.\S+', '', text)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove numbers (optional - keep for educational context)
        # text = re.sub(r'\d+', '', text)
        
        # Remove punctuation
        if self.remove_punctuation:
            text = text.translate(str.maketrans('', '', string.punctuation))
        
        if self.use_semantic_parsing and self.nlp:
            doc = self.nlp(text)
            # Combine words with their dependency tags
            semantic_tokens = []
            for token in doc:
                if token.is_punct or token.is_space:
                    continue
                word = token.lemma_ if self.lemmatize else token.text
                if self.lowercase:
                    word = word.lower()
                
                # The paper suggests adding functional dependency info
                # We'll append the dependency tag to the word
                semantic_tokens.append(f"{word}_{token.dep_}")
                # Also keep the dependency tags themselves as separate features
                semantic_tokens.append(token.dep_)
            
            return ' '.join(semantic_tokens)
        
        # Standard cleaning
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stopwords
        if self.remove_stopwords:
            tokens = [t for t in tokens if t not in self.stop_words]
        
        # Lemmatize
        if self.lemmatize:
            tokens = [self.lemmatizer.lemmatize(t) for t in tokens]
        
        # Filter by minimum word length
        tokens = [t for t in tokens if len(t) >= self.min_word_length]
        
        return ' '.join(tokens)
    
    def preprocess_dataframe(
        self,
        df: pd.DataFrame,
        text_column: str,
        label_column: str
    ) -> pd.DataFrame:
        """
        Preprocess a DataFrame containing questions and labels.
        
        Args:
            df: Input DataFrame
            text_column: Name of the column containing text
            label_column: Name of the column containing labels
            
        Returns:
            Preprocessed DataFrame with 'clean_text' column
        """
        df = df.copy()
        df['clean_text'] = df[text_column].apply(self.clean_text)
        df = df[df['clean_text'].str.len() > 0]  # Remove empty texts
        return df


def create_splits(
    df: pd.DataFrame,
    text_column: str = 'clean_text',
    label_column: str = 'label',
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split data into train, validation, and test sets with stratification.
    
    Args:
        df: Input DataFrame
        text_column: Name of text column
        label_column: Name of label column
        train_ratio: Proportion for training set
        val_ratio: Proportion for validation set
        test_ratio: Proportion for test set
        random_state: Random seed for reproducibility
        
    Returns:
        Tuple of (train_df, val_df, test_df)
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "Ratios must sum to 1.0"
    
    # First split: train vs (val + test)
    train_df, temp_df = train_test_split(
        df,
        train_size=train_ratio,
        stratify=df[label_column],
        random_state=random_state
    )
    
    # Second split: val vs test
    relative_val_ratio = val_ratio / (val_ratio + test_ratio)
    val_df, test_df = train_test_split(
        temp_df,
        train_size=relative_val_ratio,
        stratify=temp_df[label_column],
        random_state=random_state
    )
    
    return train_df, val_df, test_df


def encode_labels(
    labels: pd.Series,
    label_mapping: Optional[dict] = None
) -> Tuple[pd.Series, dict, dict]:
    """
    Encode string labels to integers.
    
    Args:
        labels: Series of string labels
        label_mapping: Optional pre-defined mapping
        
    Returns:
        Tuple of (encoded_labels, label_to_id, id_to_label)
    """
    if label_mapping is None:
        unique_labels = sorted(labels.unique())
        label_to_id = {label: idx for idx, label in enumerate(unique_labels)}
    else:
        label_to_id = label_mapping
    
    id_to_label = {v: k for k, v in label_to_id.items()}
    encoded = labels.map(label_to_id)
    
    return encoded, label_to_id, id_to_label


def load_and_preprocess_data(
    data_path: str,
    text_column: str = 'question',
    label_column: str = 'level',
    config_path: str = "config/config.yaml"
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict, dict]:
    """
    Load data, preprocess, and create splits.
    
    Args:
        data_path: Path to the raw data file (CSV)
        text_column: Name of text column in raw data
        label_column: Name of label column in raw data
        config_path: Path to configuration file
        
    Returns:
        Tuple of (train_df, val_df, test_df, label_to_id, id_to_label)
    """
    # Load config
    config = load_config(config_path)
    prep_config = config.get('preprocessing', {})
    
    # Load data
    df = pd.read_csv(data_path)
    print(f"Loaded {len(df)} samples")
    
    # Initialize preprocessor
    preprocessor = TextPreprocessor(
        lowercase=prep_config.get('lowercase', True),
        remove_punctuation=prep_config.get('remove_punctuation', True),
        remove_stopwords=prep_config.get('remove_stopwords', False),
        lemmatize=prep_config.get('lemmatize', True),
        min_word_length=prep_config.get('min_word_length', 2),
        use_semantic_parsing=prep_config.get('use_semantic_parsing', False)
    )
    
    # Preprocess
    df = preprocessor.preprocess_dataframe(df, text_column, label_column)
    df = df.rename(columns={label_column: 'label'})
    print(f"After preprocessing: {len(df)} samples")
    
    # Encode labels
    df['label_id'], label_to_id, id_to_label = encode_labels(df['label'])
    
    # Create splits
    train_df, val_df, test_df = create_splits(
        df,
        text_column='clean_text',
        label_column='label',
        train_ratio=config['data']['train_ratio'],
        val_ratio=config['data']['val_ratio'],
        test_ratio=config['data']['test_ratio']
    )
    
    print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    
    return train_df, val_df, test_df, label_to_id, id_to_label


if __name__ == "__main__":
    # Example usage
    import argparse
    import os
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, required=True, help='Path to raw data CSV')
    parser.add_argument('--text-col', type=str, default='question', help='Text column name')
    parser.add_argument('--label-col', type=str, default='level', help='Label column name')
    parser.add_argument('--output-dir', type=str, default='data/processed', help='Output directory')
    args = parser.parse_args()
    
    # Load and preprocess
    train_df, val_df, test_df, label_to_id, id_to_label = load_and_preprocess_data(
        args.data,
        text_column=args.text_col,
        label_column=args.label_col
    )
    
    # Save processed data
    os.makedirs(args.output_dir, exist_ok=True)
    train_df.to_csv(os.path.join(args.output_dir, 'train.csv'), index=False)
    val_df.to_csv(os.path.join(args.output_dir, 'val.csv'), index=False)
    test_df.to_csv(os.path.join(args.output_dir, 'test.csv'), index=False)
    
    # Save label mappings
    import json
    with open(os.path.join(args.output_dir, 'label_mapping.json'), 'w') as f:
        json.dump({'label_to_id': label_to_id, 'id_to_label': id_to_label}, f, indent=2)
    
    print(f"Saved processed data to {args.output_dir}")
