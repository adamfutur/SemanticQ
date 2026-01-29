"""
Evaluation Module for Bloom's Taxonomy Classifiers

Computes accuracy, precision, recall, F1-score, and confusion matrices.
"""

import json
import os
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


class ModelEvaluator:
    """
    Evaluator for comparing Bloom's Taxonomy classifiers.
    """
    
    def __init__(self, labels: List[str]):
        """
        Initialize evaluator.
        
        Args:
            labels: List of class labels in order
        """
        self.labels = labels
        self.results = {}
    
    def evaluate(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        model_name: str
    ) -> Dict:
        """
        Evaluate predictions and store results.
        
        Args:
            y_true: True labels (integers)
            y_pred: Predicted labels (integers)
            model_name: Name of the model for storing results
            
        Returns:
            Dictionary of metrics
        """
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision_weighted': precision_score(y_true, y_pred, average='weighted', zero_division=0),
            'recall_weighted': recall_score(y_true, y_pred, average='weighted', zero_division=0),
            'f1_weighted': f1_score(y_true, y_pred, average='weighted', zero_division=0),
            'precision_macro': precision_score(y_true, y_pred, average='macro', zero_division=0),
            'recall_macro': recall_score(y_true, y_pred, average='macro', zero_division=0),
            'f1_macro': f1_score(y_true, y_pred, average='macro', zero_division=0),
        }
        
        # Per-class metrics
        per_class_precision = precision_score(y_true, y_pred, average=None, zero_division=0)
        per_class_recall = recall_score(y_true, y_pred, average=None, zero_division=0)
        per_class_f1 = f1_score(y_true, y_pred, average=None, zero_division=0)
        
        metrics['per_class'] = {
            label: {
                'precision': float(per_class_precision[i]),
                'recall': float(per_class_recall[i]),
                'f1': float(per_class_f1[i])
            }
            for i, label in enumerate(self.labels)
        }
        
        # Confusion matrix
        metrics['confusion_matrix'] = confusion_matrix(y_true, y_pred).tolist()
        
        # Full classification report
        metrics['classification_report'] = classification_report(
            y_true, y_pred, target_names=self.labels, zero_division=0
        )
        
        self.results[model_name] = metrics
        
        return metrics
    
    def print_results(self, model_name: str) -> None:
        """Print formatted results for a model."""
        if model_name not in self.results:
            print(f"No results for {model_name}")
            return
        
        metrics = self.results[model_name]
        
        print(f"\n{'='*60}")
        print(f"Results for {model_name}")
        print('='*60)
        print(f"\nOverall Metrics:")
        print(f"  Accuracy:           {metrics['accuracy']:.4f}")
        print(f"  Precision (weighted): {metrics['precision_weighted']:.4f}")
        print(f"  Recall (weighted):    {metrics['recall_weighted']:.4f}")
        print(f"  F1-Score (weighted):  {metrics['f1_weighted']:.4f}")
        print(f"  F1-Score (macro):     {metrics['f1_macro']:.4f}")
        print(f"\nClassification Report:\n")
        print(metrics['classification_report'])
    
    def plot_confusion_matrix(
        self,
        model_name: str,
        save_path: Optional[str] = None,
        figsize: tuple = (10, 8)
    ) -> None:
        """
        Plot confusion matrix heatmap.
        
        Args:
            model_name: Name of model to plot
            save_path: Optional path to save figure
            figsize: Figure size
        """
        if model_name not in self.results:
            print(f"No results for {model_name}")
            return
        
        cm = np.array(self.results[model_name]['confusion_matrix'])
        
        plt.figure(figsize=figsize)
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=self.labels,
            yticklabels=self.labels
        )
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title(f'Confusion Matrix - {model_name}')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved confusion matrix to {save_path}")
        
        plt.show()
    
    def compare_models(
        self,
        save_path: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Create comparison table of all models.
        
        Args:
            save_path: Optional path to save comparison CSV
            
        Returns:
            DataFrame with comparison metrics
        """
        comparison_data = []
        
        for model_name, metrics in self.results.items():
            row = {
                'Model': model_name,
                'Accuracy': metrics['accuracy'],
                'Precision': metrics['precision_weighted'],
                'Recall': metrics['recall_weighted'],
                'F1-Score': metrics['f1_weighted'],
                'F1 (Macro)': metrics['f1_macro']
            }
            comparison_data.append(row)
        
        df = pd.DataFrame(comparison_data)
        
        print("\n" + "="*70)
        print("Model Comparison")
        print("="*70)
        print(df.to_string(index=False))
        
        if save_path:
            df.to_csv(save_path, index=False)
            print(f"\nSaved comparison to {save_path}")
        
        return df
    
    def plot_comparison(
        self,
        save_path: Optional[str] = None,
        figsize: tuple = (12, 6)
    ) -> None:
        """
        Plot bar chart comparing models.
        
        Args:
            save_path: Optional path to save figure
            figsize: Figure size
        """
        if len(self.results) == 0:
            print("No results to compare")
            return
        
        models = list(self.results.keys())
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        
        x = np.arange(len(metrics))
        width = 0.35 / max(1, len(models) - 1) if len(models) > 1 else 0.35
        
        fig, ax = plt.subplots(figsize=figsize)
        
        for i, model in enumerate(models):
            values = [
                self.results[model]['accuracy'],
                self.results[model]['precision_weighted'],
                self.results[model]['recall_weighted'],
                self.results[model]['f1_weighted']
            ]
            offset = (i - len(models)/2 + 0.5) * width
            ax.bar(x + offset, values, width, label=model)
        
        ax.set_xlabel('Metric')
        ax.set_ylabel('Score')
        ax.set_title('Model Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(metrics)
        ax.legend()
        ax.set_ylim(0, 1)
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved comparison plot to {save_path}")
        
        plt.show()
    
    def save_results(self, save_dir: str) -> None:
        """
        Save all results to JSON.
        
        Args:
            save_dir: Directory to save results
        """
        os.makedirs(save_dir, exist_ok=True)
        
        # Convert numpy arrays to lists for JSON serialization
        serializable_results = {}
        for model_name, metrics in self.results.items():
            serializable_results[model_name] = {
                k: v for k, v in metrics.items()
            }
        
        with open(os.path.join(save_dir, 'evaluation_results.json'), 'w') as f:
            json.dump(serializable_results, f, indent=2, default=str)
        
        print(f"Results saved to {save_dir}/evaluation_results.json")


def evaluate_fasttext_model(
    model,
    test_df: pd.DataFrame,
    evaluator: ModelEvaluator
) -> Dict:
    """
    Evaluate FastText model on test set.
    
    Args:
        model: Trained SemanticFastText model
        test_df: Test DataFrame
        evaluator: ModelEvaluator instance
        
    Returns:
        Evaluation metrics
    """
    # Get predictions
    X_test = model.get_embeddings(test_df['clean_text'].tolist())
    y_true = test_df['label_id'].values
    y_pred = model.classifier.predict(X_test)
    
    return evaluator.evaluate(y_true, y_pred, 'Semantic-FastText')


def evaluate_bert_model(
    model,
    test_df: pd.DataFrame,
    evaluator: ModelEvaluator,
    batch_size: int = 16
) -> Dict:
    """
    Evaluate BERT model on test set.
    
    Args:
        model: Trained SemanticBERT model
        test_df: Test DataFrame
        evaluator: ModelEvaluator instance
        batch_size: Batch size for evaluation
        
    Returns:
        Evaluation metrics
    """
    import torch
    
    y_true = test_df['label_id'].values
    y_pred = []
    
    model.model.eval()
    with torch.no_grad():
        for text in test_df['clean_text'].tolist():
            encoding = model.tokenizer(
                text,
                truncation=True,
                padding='max_length',
                max_length=model.max_length,
                return_tensors='pt'
            )
            
            input_ids = encoding['input_ids'].to(model.device)
            attention_mask = encoding['attention_mask'].to(model.device)
            
            outputs = model.model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            
            pred = torch.argmax(outputs.logits, dim=1).cpu().item()
            y_pred.append(pred)
    
    return evaluator.evaluate(y_true, np.array(y_pred), 'Semantic-BERT')


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate models")
    parser.add_argument('--model', type=str, choices=['fasttext', 'bert', 'both'], required=True)
    parser.add_argument('--test-data', type=str, required=True, help='Path to test CSV')
    parser.add_argument('--fasttext-dir', type=str, default='models/fasttext')
    parser.add_argument('--bert-dir', type=str, default='models/bert')
    parser.add_argument('--output-dir', type=str, default='results')
    args = parser.parse_args()
    
    # Load test data
    test_df = pd.read_csv(args.test_data)
    
    # Get labels
    labels = ["Remember", "Understand", "Apply", "Analyze", "Evaluate", "Create"]
    evaluator = ModelEvaluator(labels)
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    if args.model in ['fasttext', 'both']:
        from fasttext_classifier import SemanticFastText
        ft_model = SemanticFastText.load(args.fasttext_dir)
        evaluate_fasttext_model(ft_model, test_df, evaluator)
        evaluator.print_results('Semantic-FastText')
        evaluator.plot_confusion_matrix(
            'Semantic-FastText',
            os.path.join(args.output_dir, 'fasttext_confusion.png')
        )
    
    if args.model in ['bert', 'both']:
        from bert_classifier import SemanticBERT
        bert_model = SemanticBERT.load(args.bert_dir)
        evaluate_bert_model(bert_model, test_df, evaluator)
        evaluator.print_results('Semantic-BERT')
        evaluator.plot_confusion_matrix(
            'Semantic-BERT',
            os.path.join(args.output_dir, 'bert_confusion.png')
        )
    
    if args.model == 'both':
        evaluator.compare_models(os.path.join(args.output_dir, 'comparison.csv'))
        evaluator.plot_comparison(os.path.join(args.output_dir, 'comparison.png'))
    
    evaluator.save_results(args.output_dir)
