"""
Unified Inference Pipeline for Bloom's Taxonomy Classifiers

Provides a unified API for classifying questions using either
Semantic-FastText or Semantic-BERT models.
"""

import os
from typing import Dict, List, Optional, Union

import numpy as np


class BloomClassifier:
    """
    Unified classifier interface for Bloom's Taxonomy question classification.
    
    Supports both Semantic-FastText and Semantic-BERT models.
    Now utilizes Semantic Dependency Parsing (spaCy) to enrich 
    textual inputs with functional roles for better performance.
    
    Usage:
        # FastText (Enriched)
        classifier = BloomClassifier(model_type="fasttext", model_path="models/fasttext")
        result = classifier.predict("What is the capital of France?")
        
        # BERT (Semantic)
        classifier = BloomClassifier(model_type="bert", model_path="models/bert")
        result = classifier.predict("Design a new experiment for testing...")
    """
    
    BLOOM_LEVELS = [
        "Remember",
        "Understand", 
        "Apply",
        "Analyze",
        "Evaluate",
        "Create"
    ]
    
    LEVEL_DESCRIPTIONS = {
        "Remember": "Recall facts and basic concepts",
        "Understand": "Explain ideas or concepts",
        "Apply": "Use information in new situations",
        "Analyze": "Draw connections among ideas",
        "Evaluate": "Justify a decision or course of action",
        "Create": "Produce new or original work"
    }
    
    def __init__(
        self,
        model_type: str = "fasttext",
        model_path: Optional[str] = None,
        device: Optional[str] = None
    ):
        """
        Initialize the classifier.
        
        Args:
            model_type: Type of model ("fasttext" or "bert")
            model_path: Path to saved model directory
            device: Device for BERT model ('cuda' or 'cpu')
        """
        self.model_type = model_type.lower()
        
        if self.model_type not in ["fasttext", "bert"]:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Default paths
        if model_path is None:
            model_path = f"models/{self.model_type}"
        
        self.model_path = model_path
        self.device = device
        self.model = None
        
        self._load_model()
    
    def _load_model(self) -> None:
        """Load the appropriate model."""
        if self.model_type == "fasttext":
            from fasttext_classifier import SemanticFastText
            self.model = SemanticFastText.load(self.model_path)
        else:
            from bert_classifier import SemanticBERT
            self.model = SemanticBERT.load(self.model_path, device=self.device)
    
    def predict(
        self,
        text: Union[str, List[str]],
        return_all_probs: bool = False
    ) -> Union[Dict, List[Dict]]:
        """
        Classify educational question(s) by Bloom's Taxonomy level.
        
        Args:
            text: Single question string or list of questions
            return_all_probs: If True, include probabilities for all classes
            
        Returns:
            For single text: Dict with 'level', 'confidence', and optionally 'all_probabilities'
            For multiple texts: List of such dicts
        """
        single_input = isinstance(text, str)
        texts = [text] if single_input else text
        
        # Get predictions
        labels, probs = self.model.predict(texts)
        
        results = []
        for i, (label, prob) in enumerate(zip(labels, probs)):
            confidence = float(prob.max())
            result = {
                'level': label,
                'confidence': confidence,
                'description': self.LEVEL_DESCRIPTIONS.get(label, "")
            }
            
            if return_all_probs:
                result['all_probabilities'] = {
                    self.model.id_to_label[j]: float(p)
                    for j, p in enumerate(prob)
                }
            
            results.append(result)
        
        return results[0] if single_input else results
    
    def batch_predict(
        self,
        texts: List[str],
        batch_size: int = 32
    ) -> List[Dict]:
        """
        Classify a batch of questions efficiently.
        
        Args:
            texts: List of question strings
            batch_size: Processing batch size
            
        Returns:
            List of prediction dictionaries
        """
        all_results = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            results = self.predict(batch)
            if isinstance(results, dict):
                results = [results]
            all_results.extend(results)
        
        return all_results
    
    def explain(self, text: str) -> str:
        """
        Get detailed explanation of the classification.
        
        Args:
            text: Question to classify
            
        Returns:
            Formatted explanation string
        """
        result = self.predict(text, return_all_probs=True)
        
        explanation = f"""
Question: "{text}"

Classification: {result['level']}
Confidence: {result['confidence']:.2%}
Description: {result['description']}

Probability Distribution:
"""
        
        sorted_probs = sorted(
            result['all_probabilities'].items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        for level, prob in sorted_probs:
            bar = "█" * int(prob * 20)
            explanation += f"  {level:12} {prob:6.2%} {bar}\n"
        
        return explanation
    
    @property
    def model_info(self) -> Dict:
        """Get information about the loaded model."""
        return {
            'model_type': self.model_type,
            'model_path': self.model_path,
            'num_classes': len(self.BLOOM_LEVELS),
            'classes': self.BLOOM_LEVELS
        }


def interactive_demo():
    """Run interactive classification demo."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Bloom's Taxonomy Question Classifier")
    parser.add_argument(
        '--model', 
        type=str, 
        choices=['fasttext', 'bert'], 
        default='fasttext',
        help='Model type to use'
    )
    parser.add_argument(
        '--model-path',
        type=str,
        default=None,
        help='Path to model directory'
    )
    args = parser.parse_args()
    
    print("=" * 60)
    print("Bloom's Taxonomy Question Classifier")
    print("=" * 60)
    print(f"\nLoading {args.model.upper()} model...")
    
    try:
        classifier = BloomClassifier(
            model_type=args.model,
            model_path=args.model_path
        )
        print("Model loaded successfully!\n")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    print("Bloom's Taxonomy Levels:")
    for level, desc in classifier.LEVEL_DESCRIPTIONS.items():
        print(f"  • {level}: {desc}")
    
    print("\n" + "-" * 60)
    print("Enter questions to classify (type 'quit' to exit)")
    print("-" * 60)
    
    while True:
        try:
            question = input("\nQuestion: ").strip()
            
            if question.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break
            
            if not question:
                continue
            
            print(classifier.explain(question))
            
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break


def batch_demo():
    """Demonstrate batch classification."""
    print("Loading FastText model...")
    classifier = BloomClassifier(model_type="fasttext")
    
    sample_questions = [
        # Remember
        "What is the definition of photosynthesis?",
        "List the planets in our solar system.",
        
        # Understand
        "Explain how the water cycle works.",
        "Describe the process of cell division.",
        
        # Apply
        "Calculate the area of a triangle with base 5 and height 10.",
        "Use the Pythagorean theorem to find the hypotenuse.",
        
        # Analyze
        "Compare and contrast mitosis and meiosis.",
        "What patterns can you identify in this dataset?",
        
        # Evaluate
        "Assess the effectiveness of renewable energy sources.",
        "Critique the argument presented in this essay.",
        
        # Create
        "Design an experiment to test plant growth.",
        "Develop a new solution to reduce plastic waste."
    ]
    
    print(f"\nClassifying {len(sample_questions)} sample questions...\n")
    
    for q in sample_questions:
        result = classifier.predict(q)
        print(f"[{result['level']:10}] ({result['confidence']:.0%}) {q[:60]}...")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--batch":
        batch_demo()
    else:
        interactive_demo()
