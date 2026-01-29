"""
Semantic-FastText Classifier for Bloom's Taxonomy

This module trains FastText word embeddings, generates sentence vectors
via averaging, and trains an SVM or MLP classifier.
"""

import json
import os
import pickle
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml
from gensim.models import FastText
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from tqdm import tqdm


def load_config(config_path: str = "config/config.yaml") -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


class SemanticFastText:
    """
    Semantic-FastText model for question classification.
    
    Trains FastText embeddings and uses averaged word vectors
    as sentence representations for classification.
    """
    
    def __init__(
        self,
        vector_size: int = 100,
        window: int = 5,
        min_count: int = 1,
        epochs: int = 50,
        classifier_type: str = "svm"
    ):
        """
        Initialize Semantic-FastText model.
        
        Args:
            vector_size: Dimension of word embeddings
            window: Context window size for training
            min_count: Minimum word frequency
            epochs: Training epochs for FastText
            classifier_type: Type of classifier ("svm" or "mlp")
        """
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.epochs = epochs
        self.classifier_type = classifier_type
        
        self.fasttext_model: Optional[FastText] = None
        self.classifier = None
        self.label_to_id: Optional[Dict] = None
        self.id_to_label: Optional[Dict] = None
    
    def train_embeddings(
        self,
        sentences: List[List[str]],
        save_path: Optional[str] = None
    ) -> None:
        """
        Train FastText embeddings on tokenized sentences.
        
        Args:
            sentences: List of tokenized sentences (list of word lists)
            save_path: Optional path to save the model
        """
        print(f"Training FastText embeddings on {len(sentences)} sentences...")
        
        self.fasttext_model = FastText(
            sentences=sentences,
            vector_size=self.vector_size,
            window=self.window,
            min_count=self.min_count,
            epochs=self.epochs,
            workers=4,
            sg=1  # Skip-gram
        )
        
        print(f"Vocabulary size: {len(self.fasttext_model.wv)}")
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            self.fasttext_model.save(save_path)
            print(f"Saved FastText model to {save_path}")
    
    def load_embeddings(self, model_path: str) -> None:
        """Load pre-trained FastText embeddings."""
        self.fasttext_model = FastText.load(model_path)
        print(f"Loaded FastText model from {model_path}")
    
    def get_sentence_vector(self, sentence: str) -> np.ndarray:
        """
        Generate sentence embedding by averaging word vectors.
        
        Args:
            sentence: Preprocessed text string
            
        Returns:
            Averaged word vector as sentence representation
        """
        if self.fasttext_model is None:
            raise ValueError("FastText model not trained or loaded")
        
        words = sentence.split()
        if not words:
            return np.zeros(self.vector_size)
        
        # Get word vectors (FastText can handle OOV words)
        vectors = [self.fasttext_model.wv[word] for word in words]
        
        # Average word vectors
        return np.mean(vectors, axis=0)
    
    def get_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings for a list of texts.
        
        Args:
            texts: List of preprocessed text strings
            
        Returns:
            Array of shape (n_samples, vector_size)
        """
        embeddings = []
        for text in tqdm(texts, desc="Generating embeddings"):
            embeddings.append(self.get_sentence_vector(text))
        return np.array(embeddings)
    
    def train_classifier(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        label_to_id: Dict,
        classifier_params: Optional[Dict] = None
    ) -> None:
        """
        Train the classifier on sentence embeddings.
        
        Args:
            X_train: Training embeddings
            y_train: Training labels
            label_to_id: Label to ID mapping
            classifier_params: Optional classifier hyperparameters
        """
        self.label_to_id = label_to_id
        self.id_to_label = {v: k for k, v in label_to_id.items()}
        
        if classifier_params is None:
            classifier_params = {}
        
        if self.classifier_type == "svm":
            self.classifier = SVC(
                kernel=classifier_params.get('kernel', 'rbf'),
                C=classifier_params.get('C', 1.0),
                gamma=classifier_params.get('gamma', 'scale'),
                probability=True,
                random_state=42
            )
        elif self.classifier_type == "mlp":
            hidden_layers = classifier_params.get('hidden_layers', [256, 128])
            self.classifier = MLPClassifier(
                hidden_layer_sizes=tuple(hidden_layers),
                activation='relu',
                solver='adam',
                alpha=0.0001,
                batch_size=classifier_params.get('batch_size', 32),
                learning_rate='adaptive',
                learning_rate_init=classifier_params.get('learning_rate', 0.001),
                max_iter=classifier_params.get('epochs', 100),
                early_stopping=True,
                validation_fraction=0.1,
                random_state=42,
                verbose=True
            )
        else:
            raise ValueError(f"Unknown classifier type: {self.classifier_type}")
        
        print(f"Training {self.classifier_type.upper()} classifier...")
        self.classifier.fit(X_train, y_train)
        print("Classifier training complete!")
    
    def predict(self, texts: List[str]) -> Tuple[List[str], np.ndarray]:
        """
        Predict labels for new texts.
        
        Args:
            texts: List of preprocessed text strings
            
        Returns:
            Tuple of (predicted_labels, probabilities)
        """
        if self.classifier is None:
            raise ValueError("Classifier not trained")
        
        embeddings = self.get_embeddings(texts)
        probs = self.classifier.predict_proba(embeddings)
        
        # Determine labels from highest probability to ensure consistency
        pred_ids = np.argmax(probs, axis=1)
        pred_labels = [self.id_to_label[pid] for pid in pred_ids]
        
        return pred_labels, probs
    
    def predict_single(self, text: str) -> Dict:
        """
        Predict label for a single text with confidence.
        
        Args:
            text: Preprocessed text string
            
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
        Save the complete model (FastText + classifier).
        
        Args:
            save_dir: Directory to save model files
        """
        os.makedirs(save_dir, exist_ok=True)
        
        # Save FastText model
        if self.fasttext_model:
            self.fasttext_model.save(os.path.join(save_dir, 'fasttext.model'))
        
        # Save classifier
        if self.classifier:
            with open(os.path.join(save_dir, 'classifier.pkl'), 'wb') as f:
                pickle.dump(self.classifier, f)
        
        # Save label mappings
        with open(os.path.join(save_dir, 'label_mapping.json'), 'w') as f:
            json.dump({
                'label_to_id': self.label_to_id,
                'id_to_label': {str(k): v for k, v in self.id_to_label.items()}
            }, f, indent=2)
        
        # Save config
        with open(os.path.join(save_dir, 'model_config.json'), 'w') as f:
            json.dump({
                'vector_size': self.vector_size,
                'window': self.window,
                'min_count': self.min_count,
                'epochs': self.epochs,
                'classifier_type': self.classifier_type
            }, f, indent=2)
        
        print(f"Model saved to {save_dir}")
    
    @classmethod
    def load(cls, load_dir: str) -> 'SemanticFastText':
        """
        Load a saved model.
        
        Args:
            load_dir: Directory containing model files
            
        Returns:
            Loaded SemanticFastText instance
        """
        # Load config
        with open(os.path.join(load_dir, 'model_config.json'), 'r') as f:
            config = json.load(f)
        
        model = cls(
            vector_size=config['vector_size'],
            window=config['window'],
            min_count=config['min_count'],
            epochs=config['epochs'],
            classifier_type=config['classifier_type']
        )
        
        # Load FastText
        model.fasttext_model = FastText.load(os.path.join(load_dir, 'fasttext.model'))
        
        # Load classifier
        with open(os.path.join(load_dir, 'classifier.pkl'), 'rb') as f:
            model.classifier = pickle.load(f)
        
        # Load label mappings
        with open(os.path.join(load_dir, 'label_mapping.json'), 'r') as f:
            mappings = json.load(f)
            model.label_to_id = mappings['label_to_id']
            model.id_to_label = {int(k): v for k, v in mappings['id_to_label'].items()}
        
        print(f"Model loaded from {load_dir}")
        return model


def train_fasttext_model(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    label_to_id: Dict,
    config: Dict,
    save_dir: str = "models/fasttext"
) -> SemanticFastText:
    """
    Train complete Semantic-FastText model.
    
    Args:
        train_df: Training DataFrame with 'clean_text' and 'label_id'
        val_df: Validation DataFrame
        label_to_id: Label to ID mapping
        config: Configuration dictionary
        save_dir: Directory to save the model
        
    Returns:
        Trained SemanticFastText model
    """
    ft_config = config.get('fasttext', {})
    clf_config = config.get('classifier', {})
    
    # Initialize model
    model = SemanticFastText(
        vector_size=ft_config.get('vector_size', 100),
        window=ft_config.get('window', 5),
        min_count=ft_config.get('min_count', 1),
        epochs=ft_config.get('epochs', 50),
        classifier_type=clf_config.get('type', 'svm')
    )
    
    # Prepare sentences for FastText training
    all_texts = pd.concat([train_df['clean_text'], val_df['clean_text']])
    sentences = [text.split() for text in all_texts]
    
    # Train embeddings
    model.train_embeddings(sentences)
    
    # Generate training embeddings
    X_train = model.get_embeddings(train_df['clean_text'].tolist())
    y_train = train_df['label_id'].values
    
    # Train classifier
    clf_params = clf_config.get(clf_config.get('type', 'svm'), {})
    model.train_classifier(X_train, y_train, label_to_id, clf_params)
    
    # Save model
    model.save(save_dir)
    
    return model


if __name__ == "__main__":
    import argparse
    from preprocessing import load_and_preprocess_data
    
    parser = argparse.ArgumentParser(description="Train Semantic-FastText classifier")
    parser.add_argument('--data', type=str, required=True, help='Path to raw data CSV')
    parser.add_argument('--text-col', type=str, default='question', help='Text column')
    parser.add_argument('--label-col', type=str, default='level', help='Label column')
    parser.add_argument('--config', type=str, default='config/config.yaml', help='Config path')
    parser.add_argument('--output-dir', type=str, default='models/fasttext', help='Output dir')
    parser.add_argument('--mode', type=str, choices=['train', 'predict'], default='train')
    args = parser.parse_args()
    
    config = load_config(args.config)
    
    if args.mode == 'train':
        # Load and preprocess data
        train_df, val_df, test_df, label_to_id, id_to_label = load_and_preprocess_data(
            args.data,
            text_column=args.text_col,
            label_column=args.label_col,
            config_path=args.config
        )
        
        # Train model
        model = train_fasttext_model(
            train_df, val_df, label_to_id, config, args.output_dir
        )
        
        # Quick evaluation on validation set
        X_val = model.get_embeddings(val_df['clean_text'].tolist())
        y_val = val_df['label_id'].values
        val_acc = model.classifier.score(X_val, y_val)
        print(f"\nValidation Accuracy: {val_acc:.4f}")
    
    else:
        # Prediction mode
        model = SemanticFastText.load(args.output_dir)
        
        # Example predictions
        test_questions = [
            "What is the capital of France?",
            "Explain how photosynthesis works.",
            "Design a new experiment to test plant growth."
        ]
        
        for q in test_questions:
            result = model.predict_single(q.lower())
            print(f"\nQuestion: {q}")
            print(f"  Level: {result['level']} (confidence: {result['confidence']:.2f})")
