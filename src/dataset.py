"""
PyTorch Dataset classes for BERT and FastText models.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class BloomDataset(Dataset):
    """
    Generic dataset for Bloom's Taxonomy classification.
    
    Works with both FastText embeddings (pre-computed) and raw text for BERT.
    """
    
    def __init__(
        self,
        texts: List[str],
        labels: List[int],
        embeddings: Optional[np.ndarray] = None
    ):
        """
        Initialize dataset.
        
        Args:
            texts: List of preprocessed text strings
            labels: List of integer label IDs
            embeddings: Optional pre-computed embeddings (for FastText)
        """
        self.texts = texts
        self.labels = labels
        self.embeddings = embeddings
    
    def __len__(self) -> int:
        return len(self.texts)
    
    def __getitem__(self, idx: int) -> Dict:
        item = {
            'text': self.texts[idx],
            'label': torch.tensor(self.labels[idx], dtype=torch.long)
        }
        
        if self.embeddings is not None:
            item['embedding'] = torch.tensor(
                self.embeddings[idx], dtype=torch.float32
            )
        
        return item


class BertDataset(Dataset):
    """
    Dataset for BERT model with tokenization.
    """
    
    def __init__(
        self,
        texts: List[str],
        labels: List[int],
        tokenizer,
        max_length: int = 128
    ):
        """
        Initialize BERT dataset with tokenizer.
        
        Args:
            texts: List of preprocessed text strings
            labels: List of integer label IDs
            tokenizer: HuggingFace tokenizer
            max_length: Maximum sequence length
        """
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self) -> int:
        return len(self.texts)
    
    def __getitem__(self, idx: int) -> Dict:
        text = self.texts[idx]
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'label': torch.tensor(label, dtype=torch.long)
        }


def create_dataloaders(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    tokenizer=None,
    batch_size: int = 16,
    max_length: int = 128,
    embeddings: Optional[Dict[str, np.ndarray]] = None,
    num_workers: int = 0
) -> Tuple:
    """
    Create DataLoaders for training, validation, and testing.
    
    Args:
        train_df: Training DataFrame with 'clean_text' and 'label_id'
        val_df: Validation DataFrame
        test_df: Test DataFrame
        tokenizer: BERT tokenizer (if using BERT)
        batch_size: Batch size
        max_length: Max sequence length for BERT
        embeddings: Dict of pre-computed embeddings for FastText
        num_workers: Number of data loading workers
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    from torch.utils.data import DataLoader
    
    if tokenizer is not None:
        # BERT mode
        train_dataset = BertDataset(
            train_df['clean_text'].tolist(),
            train_df['label_id'].tolist(),
            tokenizer,
            max_length
        )
        val_dataset = BertDataset(
            val_df['clean_text'].tolist(),
            val_df['label_id'].tolist(),
            tokenizer,
            max_length
        )
        test_dataset = BertDataset(
            test_df['clean_text'].tolist(),
            test_df['label_id'].tolist(),
            tokenizer,
            max_length
        )
    else:
        # FastText mode (with pre-computed embeddings)
        train_dataset = BloomDataset(
            train_df['clean_text'].tolist(),
            train_df['label_id'].tolist(),
            embeddings.get('train') if embeddings else None
        )
        val_dataset = BloomDataset(
            val_df['clean_text'].tolist(),
            val_df['label_id'].tolist(),
            embeddings.get('val') if embeddings else None
        )
        test_dataset = BloomDataset(
            test_df['clean_text'].tolist(),
            test_df['label_id'].tolist(),
            embeddings.get('test') if embeddings else None
        )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )
    
    return train_loader, val_loader, test_loader
