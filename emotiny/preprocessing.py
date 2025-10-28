import re
import numpy as np
import pandas as pd
from typing import List, Tuple, Optional
from sentence_transformers import SentenceTransformer
import torch
from .config import EMBEDDING_MODEL, EMOTION_LABELS


class EmoTinyPreprocessor:
    def __init__(self, model_name: str = EMBEDDING_MODEL, device: str = "cuda"):
        """Initialize the preprocessor with a sentence transformer model"""
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
        """Clean text to handle ASR noise and normalize input"""
        if pd.isna(text) or text is None:
            return ""
        text = text.strip()  # Handle common ASR artifacts
        text = re.sub(r'\s+', ' ', text) # Remove excessive whitespace
        text = re.sub(r'[.]{2,}', '.', text)  # Multiple dots
        text = re.sub(r'[?]{2,}', '?', text)  # Multiple question marks
        text = re.sub(r'[!]{2,}', '!', text)  # Multiple exclamation marks
        text = re.sub(r'[^\w\s.,!?¿¡áéíóúàèìòùâêîôûãõçñü-]', '', text, flags=re.IGNORECASE)  # Remove or normalize special characters that might confuse the model
        if len(text.strip()) < 2:
            return ""
        return text.strip()
    
    def encode_texts(self, texts: List[str], batch_size: int = 32, show_progress: bool = True) -> np.ndarray:
        """Generate embeddings for a list of texts"""
        self.load_model()
        cleaned_texts = [self.clean_text(text) for text in texts]
        embeddings = self.model.encode(cleaned_texts, batch_size=batch_size, show_progress_bar=show_progress, convert_to_numpy=True, normalize_embeddings=True)
        nan_mask = np.isnan(embeddings).any(axis=1)
        if nan_mask.any():
            nan_count = nan_mask.sum()
            print(f"Warning: Found {nan_count} embeddings with NaN values. These will be replaced with zero vectors.")
            embeddings[nan_mask] = 0.0
        return embeddings
    
    def encode_single_text(self, text: str) -> np.ndarray:
        """Generate embedding for a single text (optimized for inference)"""
        self.load_model()
        cleaned_text = self.clean_text(text)
        with torch.no_grad():
            embedding = self.model.encode([cleaned_text], batch_size=1, show_progress_bar=False, convert_to_numpy=True, normalize_embeddings=True)
        if np.isnan(embedding).any():
            print(f"Warning: NaN values found in embedding for text: '{text[:50]}...'. Replacing with zero vector.")
            embedding = np.zeros_like(embedding)
        return embedding[0]  # Return single embedding
    
    def prepare_training_data(self, texts: List[str], labels: List[str], validation_split: float = 0.0) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
        """Prepare training data with embeddings and encoded labels"""
        print("Generating embeddings for training data...")
        X = self.encode_texts(texts, show_progress=True)
        y = np.array([self.label_to_idx.get(label, 0) for label in labels])
        valid_mask = np.isfinite(X).all(axis=1)
        if not valid_mask.all():
            invalid_count = (~valid_mask).sum()
            print(f"Warning: Filtering out {invalid_count} samples with invalid embeddings (NaN/inf)")
            X = X[valid_mask]
            y = y[valid_mask]
        if validation_split > 0:
            from sklearn.model_selection import train_test_split
            X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=validation_split, random_state=42, stratify=y)
            return X_train, y_train, X_val, y_val
        return X, y, None, None
    
    def load_dataset_from_csv(self, csv_path: str, text_column: str = "text", label_column: str = "emotion") -> Tuple[List[str], List[str]]:
        """Load dataset from CSV file"""
        df = pd.read_csv(csv_path)
        if text_column not in df.columns or label_column not in df.columns:
            raise ValueError(f"CSV must contain '{text_column}' and '{label_column}' columns")
        initial_count = len(df)
        df = df.dropna(subset=[text_column, label_column])
        df = df[df[text_column].astype(str).str.strip() != ""]
        valid_emotions = set(EMOTION_LABELS)
        df = df[df[label_column].isin(valid_emotions)]
        if len(df) == 0:
            raise ValueError(f"No valid emotions found. Expected one of: {EMOTION_LABELS}")
        filtered_count = initial_count - len(df)
        if filtered_count > 0:
            print(f"Filtered out {filtered_count} rows with missing/invalid text or labels")
        print(f"Loaded {len(df)} samples from {csv_path}")
        print(f"Emotion distribution:\n{df[label_column].value_counts()}")
        return df[text_column].tolist(), df[label_column].tolist()
    
    def get_embedding_dim(self) -> int:
        """Get the dimension of embeddings from the model."""
        self.load_model()
        return self.model.get_sentence_embedding_dimension()
    
    def validate_labels(self, labels: List[str]) -> List[str]:
        """Validate and filter emotion labels"""
        valid_labels = []
        invalid_count = 0
        for label in labels:
            if label in self.label_to_idx:
                valid_labels.append(label)
            else:
                valid_labels.append("ERROR")  # Default fallback
                invalid_count += 1
        if invalid_count:
            print(f"Warning: {invalid_count} invalid labels found")
        return valid_labels