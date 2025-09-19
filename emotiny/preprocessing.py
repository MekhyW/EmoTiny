"""
Data preprocessing and embedding generation for EmoTiny.
"""

import re
import numpy as np
import pandas as pd
from typing import List, Tuple, Optional
from sentence_transformers import SentenceTransformer
import torch
from .config import EMBEDDING_MODEL, EMOTION_LABELS


class EmoTinyPreprocessor:
    """
    Preprocessor for text data and embedding generation.
    Optimized for multilingual text and robust to ASR noise.
    """
    
    def __init__(self, model_name: str = EMBEDDING_MODEL, device: str = "cpu"):
        """
        Initialize the preprocessor with a sentence transformer model.
        
        Args:
            model_name: Name of the sentence transformer model
            device: Device to run the model on ("cpu" or "cuda")
        """
        self.model_name = model_name
        self.device = device
        self.model = None
        self.label_to_idx = {label: idx for idx, label in enumerate(EMOTION_LABELS)}
        self.idx_to_label = {idx: label for idx, label in enumerate(EMOTION_LABELS)}
        
    def load_model(self):
        """Load the sentence transformer model."""
        if self.model is None:
            print(f"Loading embedding model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name, device=self.device)
            self.model.eval()
            if self.device == "cpu":
                torch.set_num_threads(1)  # Single thread for consistent latency
                
    def clean_text(self, text: str) -> str:
        """
        Clean text to handle ASR noise and normalize input.
        
        Args:
            text: Raw text input (potentially from ASR)
            
        Returns:
            Cleaned text
        """
        text = text.strip()  # Handle common ASR artifacts
        text = re.sub(r'\s+', ' ', text) # Remove excessive whitespace
        text = re.sub(r'[.]{2,}', '.', text)  # Multiple dots
        text = re.sub(r'[?]{2,}', '?', text)  # Multiple question marks
        text = re.sub(r'[!]{2,}', '!', text)  # Multiple exclamation marks
        text = re.sub(r'[^\w\s.,!?¿¡áéíóúàèìòùâêîôûãõçñü-]', '', text, flags=re.IGNORECASE)  # Remove or normalize special characters that might confuse the model
        if len(text.strip()) < 2:
            return "neutral"  # Fallback for very short/empty text
        return text.strip()
    
    def encode_texts(self, texts: List[str], batch_size: int = 32, show_progress: bool = True) -> np.ndarray:
        """
        Generate embeddings for a list of texts.
        
        Args:
            texts: List of text strings
            batch_size: Batch size for encoding
            show_progress: Whether to show progress bar
            
        Returns:
            Numpy array of embeddings
        """
        self.load_model()
        cleaned_texts = [self.clean_text(text) for text in texts]
        embeddings = self.model.encode(
            cleaned_texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            normalize_embeddings=True  # L2 normalization for better classification
        )
        return embeddings
    
    def encode_single_text(self, text: str) -> np.ndarray:
        """
        Generate embedding for a single text (optimized for inference).
        
        Args:
            text: Input text string
            
        Returns:
            Numpy array embedding
        """
        self.load_model()
        cleaned_text = self.clean_text(text)
        with torch.no_grad():
            embedding = self.model.encode(
                [cleaned_text],
                batch_size=1,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True
            )
        return embedding[0]  # Return single embedding
    
    def prepare_training_data(self, texts: List[str], labels: List[str], validation_split: float = 0.0) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Prepare training data with embeddings and encoded labels.
        
        Args:
            texts: List of text strings
            labels: List of emotion labels
            validation_split: Fraction of data to use for validation
            
        Returns:
            Tuple of (X_train, y_train, X_val, y_val) or (X, y, None, None) if no validation split
        """
        print("Generating embeddings for training data...")
        X = self.encode_texts(texts, show_progress=True)
        y = np.array([self.label_to_idx.get(label, 0) for label in labels])
        if validation_split > 0:
            from sklearn.model_selection import train_test_split
            X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=validation_split, random_state=42, stratify=y)
            return X_train, y_train, X_val, y_val
        return X, y, None, None
    
    def load_dataset_from_csv(self, csv_path: str, text_column: str = "text", label_column: str = "emotion") -> Tuple[List[str], List[str]]:
        """
        Load dataset from CSV file.
        
        Args:
            csv_path: Path to CSV file
            text_column: Name of the text column
            label_column: Name of the label column
            
        Returns:
            Tuple of (texts, labels)
        """
        df = pd.read_csv(csv_path)
        if text_column not in df.columns or label_column not in df.columns:
            raise ValueError(f"CSV must contain '{text_column}' and '{label_column}' columns")
        valid_emotions = set(EMOTION_LABELS)
        df = df[df[label_column].isin(valid_emotions)]
        if len(df) == 0:
            raise ValueError(f"No valid emotions found. Expected one of: {EMOTION_LABELS}")
        print(f"Loaded {len(df)} samples from {csv_path}")
        print(f"Emotion distribution:\n{df[label_column].value_counts()}")
        return df[text_column].tolist(), df[label_column].tolist()
    
    def get_embedding_dim(self) -> int:
        """Get the dimension of embeddings from the model."""
        self.load_model()
        return self.model.get_sentence_embedding_dimension()
    
    def validate_labels(self, labels: List[str]) -> List[str]:
        """
        Validate and filter emotion labels.
        
        Args:
            labels: List of emotion labels
            
        Returns:
            List of valid emotion labels
        """
        valid_labels = []
        invalid_count = 0
        for label in labels:
            if label in self.label_to_idx:
                valid_labels.append(label)
            else:
                valid_labels.append("neutral")  # Default fallback
                invalid_count += 1
        if invalid_count > 0:
            print(f"Warning: {invalid_count} invalid labels found, replaced with 'neutral'")
        return valid_labels