"""
Semantic-BERT Classifier for Bloom's Taxonomy

Fine-tunes a pre-trained BERT model for multi-class classification
of educational questions into Bloom's cognitive levels.
"""

# Disable TensorFlow to avoid ml_dtypes conflict
import os
os.environ['USE_TF'] = '0'
os.environ['USE_TORCH'] = '1'
os.environ['TRANSFORMERS_NO_TF'] = '1'

import json
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    BertForSequenceClassification,
    BertTokenizer,
    get_linear_schedule_with_warmup,
)

from dataset import BertDataset


def load_config(config_path: str = "config/config.yaml") -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


class SemanticBERT:
    """
    Semantic-BERT model for question classification.
    
    Fine-tunes BERT for multi-class classification using [CLS] token.
    """
    
    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        num_labels: int = 6,
        max_length: int = 128,
        device: Optional[str] = None
    ):
        """
        Initialize Semantic-BERT model.
        
        Args:
            model_name: HuggingFace model name
            num_labels: Number of classification labels
            max_length: Maximum sequence length
            device: Device to use ('cuda' or 'cpu')
        """
        self.model_name = model_name
        self.num_labels = num_labels
        self.max_length = max_length
        
        # Set device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        print(f"Using device: {self.device}")
        
        # Load tokenizer and model
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels
        )
        self.model.to(self.device)
        
        self.label_to_id: Optional[Dict] = None
        self.id_to_label: Optional[Dict] = None
    
    def create_dataloader(
        self,
        texts: List[str],
        labels: List[int],
        batch_size: int = 16,
        shuffle: bool = True
    ) -> DataLoader:
        """
        Create a DataLoader for training/evaluation.
        
        Args:
            texts: List of text strings
            labels: List of integer labels
            batch_size: Batch size
            shuffle: Whether to shuffle data
            
        Returns:
            PyTorch DataLoader
        """
        dataset = BertDataset(
            texts=texts,
            labels=labels,
            tokenizer=self.tokenizer,
            max_length=self.max_length
        )
        
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=0
        )
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 5,
        learning_rate: float = 2e-5,
        warmup_ratio: float = 0.1,
        weight_decay: float = 0.01,
        save_dir: Optional[str] = None
    ) -> Dict:
        """
        Fine-tune BERT model.
        
        Args:
            train_loader: Training DataLoader
            val_loader: Validation DataLoader
            epochs: Number of training epochs
            learning_rate: Learning rate
            warmup_ratio: Warmup ratio for scheduler
            weight_decay: Weight decay for optimizer
            save_dir: Directory to save best model
            
        Returns:
            Training history dictionary
        """
        # Setup optimizer
        optimizer = AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # Setup scheduler
        total_steps = len(train_loader) * epochs
        warmup_steps = int(total_steps * warmup_ratio)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )
        
        # Training loop
        history = {'train_loss': [], 'val_loss': [], 'val_accuracy': []}
        best_val_acc = 0.0
        
        for epoch in range(epochs):
            # Training phase
            self.model.train()
            train_loss = 0.0
            train_steps = 0
            
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
            for batch in pbar:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['label'].to(self.device)
                
                optimizer.zero_grad()
                
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                loss = outputs.loss
                loss.backward()
                
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                
                optimizer.step()
                scheduler.step()
                
                train_loss += loss.item()
                train_steps += 1
                pbar.set_postfix({'loss': loss.item()})
            
            avg_train_loss = train_loss / train_steps
            
            # Validation phase
            val_loss, val_acc = self.evaluate(val_loader)
            
            history['train_loss'].append(avg_train_loss)
            history['val_loss'].append(val_loss)
            history['val_accuracy'].append(val_acc)
            
            print(f"\nEpoch {epoch+1}/{epochs}")
            print(f"  Train Loss: {avg_train_loss:.4f}")
            print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            
            # Save best model
            if val_acc > best_val_acc and save_dir:
                best_val_acc = val_acc
                self.save(save_dir)
                print(f"  Saved best model (acc: {val_acc:.4f})")
        
        return history
    
    def evaluate(self, dataloader: DataLoader) -> Tuple[float, float]:
        """
        Evaluate model on a dataset.
        
        Args:
            dataloader: DataLoader for evaluation
            
        Returns:
            Tuple of (loss, accuracy)
        """
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['label'].to(self.device)
                
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                total_loss += outputs.loss.item()
                
                preds = torch.argmax(outputs.logits, dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        
        avg_loss = total_loss / len(dataloader)
        accuracy = correct / total
        
        return avg_loss, accuracy
    
    def predict(self, texts: List[str]) -> Tuple[List[str], np.ndarray]:
        """
        Predict labels for new texts.
        
        Args:
            texts: List of text strings
            
        Returns:
            Tuple of (predicted_labels, probabilities)
        """
        self.model.eval()
        
        all_probs = []
        
        with torch.no_grad():
            for text in texts:
                encoding = self.tokenizer(
                    text,
                    truncation=True,
                    padding='max_length',
                    max_length=self.max_length,
                    return_tensors='pt'
                )
                
                input_ids = encoding['input_ids'].to(self.device)
                attention_mask = encoding['attention_mask'].to(self.device)
                
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                
                probs = torch.softmax(outputs.logits, dim=1)
                all_probs.append(probs.cpu().numpy()[0])
        
        all_probs = np.array(all_probs)
        all_preds = np.argmax(all_probs, axis=1)
        pred_labels = [self.id_to_label[p] for p in all_preds]
        
        return pred_labels, all_probs
    
    def predict_single(self, text: str) -> Dict:
        """
        Predict label for a single text with confidence.
        
        Args:
            text: Input text string
            
        Returns:
            Dict with 'level' and 'confidence'
        """
        labels, probs = self.predict([text])
        prob_dist = probs[0]
        max_idx = np.argmax(prob_dist)
        
        return {
            'level': self.id_to_label[max_idx],
            'confidence': float(prob_dist[max_idx]),
            'all_probabilities': {
                self.id_to_label[i]: float(p) 
                for i, p in enumerate(prob_dist)
            }
        }
    
    def save(self, save_dir: str) -> None:
        """
        Save the model and tokenizer.
        
        Args:
            save_dir: Directory to save model files
        """
        os.makedirs(save_dir, exist_ok=True)
        
        self.model.save_pretrained(save_dir)
        self.tokenizer.save_pretrained(save_dir)
        
        # Save label mappings
        if self.label_to_id:
            with open(os.path.join(save_dir, 'label_mapping.json'), 'w') as f:
                json.dump({
                    'label_to_id': self.label_to_id,
                    'id_to_label': {str(k): v for k, v in self.id_to_label.items()}
                }, f, indent=2)
        
        # Save config
        with open(os.path.join(save_dir, 'model_config.json'), 'w') as f:
            json.dump({
                'model_name': self.model_name,
                'num_labels': self.num_labels,
                'max_length': self.max_length
            }, f, indent=2)
        
        print(f"Model saved to {save_dir}")
    
    @classmethod
    def load(cls, load_dir: str, device: Optional[str] = None) -> 'SemanticBERT':
        """
        Load a saved model.
        
        Args:
            load_dir: Directory containing model files
            device: Device to load model on
            
        Returns:
            Loaded SemanticBERT instance
        """
        # Load config
        with open(os.path.join(load_dir, 'model_config.json'), 'r') as f:
            config = json.load(f)
        
        model = cls(
            model_name=load_dir,  # Load from saved directory
            num_labels=config['num_labels'],
            max_length=config['max_length'],
            device=device
        )
        
        # Load label mappings
        label_path = os.path.join(load_dir, 'label_mapping.json')
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                mappings = json.load(f)
                model.label_to_id = mappings['label_to_id']
                model.id_to_label = {int(k): v for k, v in mappings['id_to_label'].items()}
        
        print(f"Model loaded from {load_dir}")
        return model


def train_bert_model(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    label_to_id: Dict,
    config: Dict,
    save_dir: str = "models/bert"
) -> SemanticBERT:
    """
    Train complete Semantic-BERT model.
    
    Args:
        train_df: Training DataFrame with 'clean_text' and 'label_id'
        val_df: Validation DataFrame
        label_to_id: Label to ID mapping
        config: Configuration dictionary
        save_dir: Directory to save the model
        
    Returns:
        Trained SemanticBERT model
    """
    bert_config = config.get('bert', {})
    
    # Initialize model
    model = SemanticBERT(
        model_name=bert_config.get('model_name', 'bert-base-uncased'),
        num_labels=len(label_to_id),
        max_length=bert_config.get('max_length', 128)
    )
    
    model.label_to_id = label_to_id
    model.id_to_label = {v: k for k, v in label_to_id.items()}
    
    # Create dataloaders
    train_loader = model.create_dataloader(
        train_df['clean_text'].tolist(),
        train_df['label_id'].tolist(),
        batch_size=bert_config.get('batch_size', 16),
        shuffle=True
    )
    
    val_loader = model.create_dataloader(
        val_df['clean_text'].tolist(),
        val_df['label_id'].tolist(),
        batch_size=bert_config.get('batch_size', 16),
        shuffle=False
    )
    
    # Train model
    history = model.train(
        train_loader,
        val_loader,
        epochs=int(bert_config.get('epochs', 5)),
        learning_rate=float(bert_config.get('learning_rate', 2e-5)),
        warmup_ratio=float(bert_config.get('warmup_ratio', 0.1)),
        weight_decay=float(bert_config.get('weight_decay', 0.01)),
        save_dir=save_dir
    )
    
    return model


if __name__ == "__main__":
    import argparse
    from preprocessing import load_and_preprocess_data
    
    parser = argparse.ArgumentParser(description="Train Semantic-BERT classifier")
    parser.add_argument('--data', type=str, required=True, help='Path to raw data CSV')
    parser.add_argument('--text-col', type=str, default='question', help='Text column')
    parser.add_argument('--label-col', type=str, default='level', help='Label column')
    parser.add_argument('--config', type=str, default='config/config.yaml', help='Config path')
    parser.add_argument('--output-dir', type=str, default='models/bert', help='Output dir')
    parser.add_argument('--mode', type=str, choices=['train', 'predict'], default='train')
    parser.add_argument('--epochs', type=int, default=None, help='Override epochs')
    parser.add_argument('--subset', type=int, default=None, help='Use subset for testing')
    args = parser.parse_args()
    
    config = load_config(args.config)
    
    if args.epochs:
        config['bert']['epochs'] = args.epochs
    
    if args.mode == 'train':
        # Load and preprocess data
        train_df, val_df, test_df, label_to_id, id_to_label = load_and_preprocess_data(
            args.data,
            text_column=args.text_col,
            label_column=args.label_col,
            config_path=args.config
        )
        
        # Subset for testing
        if args.subset:
            train_df = train_df.head(args.subset)
            val_df = val_df.head(args.subset // 4)
        
        # Train model
        model = train_bert_model(
            train_df, val_df, label_to_id, config, args.output_dir
        )
        
        print("\nTraining complete!")
    
    else:
        # Prediction mode
        model = SemanticBERT.load(args.output_dir)
        
        # Example predictions
        test_questions = [
            "What is the capital of France?",
            "Explain how photosynthesis works.",
            "Design a new experiment to test plant growth."
        ]
        
        for q in test_questions:
            result = model.predict_single(q)
            print(f"\nQuestion: {q}")
            print(f"  Level: {result['level']} (confidence: {result['confidence']:.2f})")
