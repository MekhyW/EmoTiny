"""
Fast inference module for EmoTiny emotion classification.
"""

import os
import numpy as np
import joblib
import warnings
from typing import Optional, Union, List, Dict
import onnxruntime as ort
from .preprocessing import EmoTinyPreprocessor
from .config import EMOTION_LABELS


class EmoTinyClassifier:
    """
    Fast emotion classifier for real-time inference.
    Optimized for low latency and small footprint.
    """
    
    def __init__(self, model_path: str, use_onnx: bool = True, device: str = "cpu"):
        """
        Initialize the classifier for inference.
        
        Args:
            model_path: Path to the trained model directory
            use_onnx: Whether to use ONNX runtime for faster inference
            device: Device to run inference on ("cpu" or "cuda")
        """
        self.model_path = model_path
        self.use_onnx = use_onnx
        self.device = device
        self.classifier = None
        self.onnx_session = None
        self.preprocessor = None
        self.metadata = None
        self._load_model()
        
    def _load_model(self):
        """Load the trained model and metadata."""
        metadata_path = os.path.join(self.model_path, "metadata.joblib")
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
        self.metadata = joblib.load(metadata_path)
        embedding_model = self.metadata.get("embedding_model", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
        self.preprocessor = EmoTinyPreprocessor(embedding_model, device=self.device)
        if self.use_onnx:
            onnx_path = os.path.join(self.model_path, "classifier.onnx")
            if os.path.exists(onnx_path):
                self._load_onnx_model(onnx_path)
            else:
                warnings.warn("ONNX model not found, falling back to scikit-learn model")
                self._load_sklearn_model()
        else:
            self._load_sklearn_model()
    
    def _load_sklearn_model(self):
        """Load scikit-learn classifier."""
        classifier_path = os.path.join(self.model_path, "classifier.joblib")
        if not os.path.exists(classifier_path):
            raise FileNotFoundError(f"Classifier file not found: {classifier_path}")
        self.classifier = joblib.load(classifier_path)
        print(f"Loaded scikit-learn classifier: {type(self.classifier).__name__}")
    
    def _load_onnx_model(self, onnx_path: str):
        """Load ONNX model for faster inference."""
        try:
            providers = ["CPUExecutionProvider"]
            if self.device == "cuda":
                providers.insert(0, "CUDAExecutionProvider")
            self.onnx_session = ort.InferenceSession(onnx_path, providers=providers)
            print(f"Loaded ONNX model from: {onnx_path}")
        except Exception as e:
            warnings.warn(f"Failed to load ONNX model: {e}. Falling back to scikit-learn.")
            self._load_sklearn_model()
    
    def predict(self, text: str) -> str:
        """
        Predict emotion for a single text.
        
        Args:
            text: Input text string
            
        Returns:
            Predicted emotion label
        """
        embedding = self.preprocessor.encode_single_text(text)
        if self.onnx_session is not None:
            prediction = self._predict_onnx(embedding)
        else:
            prediction = self._predict_sklearn(embedding)
        return EMOTION_LABELS[prediction]
    
    def predict_proba(self, text: str) -> Dict[str, float]:
        """
        Predict emotion probabilities for a single text.
        
        Args:
            text: Input text string
            
        Returns:
            Dictionary mapping emotion labels to probabilities
        """
        embedding = self.preprocessor.encode_single_text(text)
        if self.onnx_session is not None:
            probabilities = self._predict_proba_onnx(embedding)
        else:
            probabilities = self._predict_proba_sklearn(embedding)
        return {label: float(prob) for label, prob in zip(EMOTION_LABELS, probabilities)}
    
    def predict_batch(self, texts: List[str]) -> List[str]:
        """
        Predict emotions for a batch of texts.
        
        Args:
            texts: List of text strings
            
        Returns:
            List of predicted emotion labels
        """
        embeddings = self.preprocessor.encode_texts(texts, show_progress=False)
        if self.onnx_session is not None:
            predictions = [self._predict_onnx(emb) for emb in embeddings]
        else:
            predictions = self.classifier.predict(embeddings)
        return [EMOTION_LABELS[pred] for pred in predictions]
    
    def _predict_sklearn(self, embedding: np.ndarray) -> int:
        """Predict using scikit-learn model."""
        prediction = self.classifier.predict(embedding.reshape(1, -1))
        return int(prediction[0])
    
    def _predict_proba_sklearn(self, embedding: np.ndarray) -> np.ndarray:
        """Predict probabilities using scikit-learn model."""
        if hasattr(self.classifier, 'predict_proba'):
            probabilities = self.classifier.predict_proba(embedding.reshape(1, -1))
            return probabilities[0]
        else:
            prediction = self._predict_sklearn(embedding)
            proba = np.zeros(len(EMOTION_LABELS))
            proba[prediction] = 1.0
            return proba
    
    def _predict_onnx(self, embedding: np.ndarray) -> int:
        """Predict using ONNX model."""
        input_name = self.onnx_session.get_inputs()[0].name
        outputs = self.onnx_session.run(None, {input_name: embedding.reshape(1, -1).astype(np.float32)})
        if len(outputs) == 1:
            return int(outputs[0][0])
        else:
            return int(np.argmax(outputs[1][0]))  # outputs[1] is usually probabilities
    
    def _predict_proba_onnx(self, embedding: np.ndarray) -> np.ndarray:
        """Predict probabilities using ONNX model."""
        input_name = self.onnx_session.get_inputs()[0].name
        outputs = self.onnx_session.run(None, {input_name: embedding.reshape(1, -1).astype(np.float32)})
        if len(outputs) > 1:
            return outputs[1][0].astype(np.float64) # Probability output available
        else:
            # Only class output, create one-hot
            prediction = int(outputs[0][0])
            proba = np.zeros(len(EMOTION_LABELS))
            proba[prediction] = 1.0
            return proba


# Global classifier instance for the simple API
_global_classifier: Optional[EmoTinyClassifier] = None


def load_model(model_path: str, use_onnx: bool = True, device: str = "cpu"):
    """
    Load a trained EmoTiny model for inference.
    
    Args:
        model_path: Path to the trained model directory
        use_onnx: Whether to use ONNX runtime for faster inference
        device: Device to run inference on ("cpu" or "cuda")
    """
    global _global_classifier
    _global_classifier = EmoTinyClassifier(model_path, use_onnx=use_onnx, device=device)
    print(f"EmoTiny model loaded from: {model_path}")


def classify_emotion(text: str) -> str:
    """
    Classify emotion for a single text using the loaded model.
    
    This is the main inference function for real-time emotion classification.
    
    Args:
        text: Input text string (1-4 sentences, multilingual support)
        
    Returns:
        Predicted emotion label (one of: neutral, happy, sad, angry, surprised, 
        disgusted, mischievous, love, nightmare)
        
    Example:
        >>> load_model("path/to/trained/model")
        >>> emotion = classify_emotion("I'm so happy today!")
        >>> print(emotion)  # "happy"
    """
    if _global_classifier is None:
        raise RuntimeError("No model loaded. Call load_model() first.")
    return _global_classifier.predict(text)


def classify_emotion_with_confidence(text: str) -> Dict[str, Union[str, float, Dict[str, float]]]:
    """
    Classify emotion with confidence scores.
    
    Args:
        text: Input text string
        
    Returns:
        Dictionary with prediction, confidence, and all probabilities
    """
    if _global_classifier is None:
        raise RuntimeError("No model loaded. Call load_model() first.")
    probabilities = _global_classifier.predict_proba(text)
    prediction = max(probabilities, key=probabilities.get)
    confidence = probabilities[prediction]
    return {
        "emotion": prediction,
        "confidence": confidence,
        "probabilities": probabilities
    }


def classify_emotions_batch(texts: List[str]) -> List[str]:
    """
    Classify emotions for a batch of texts.
    
    Args:
        texts: List of text strings
        
    Returns:
        List of predicted emotion labels
    """
    if _global_classifier is None:
        raise RuntimeError("No model loaded. Call load_model() first.")
    return _global_classifier.predict_batch(texts)


def get_supported_emotions() -> List[str]:
    """
    Get the list of supported emotion labels.
    
    Returns:
        List of emotion labels
    """
    return EMOTION_LABELS.copy()


def is_model_loaded() -> bool:
    """
    Check if a model is currently loaded.
    
    Returns:
        True if model is loaded, False otherwise
    """
    return _global_classifier is not None


def quick_test():
    """Quick test function to verify the inference pipeline."""
    if not is_model_loaded():
        print("No model loaded. Please load a model first with load_model()")
        return
    test_texts = [
        "I'm so happy today!",
        "This is terrible, I hate it.",
        "I love you so much.",
        "What a surprise!",
        "I'm feeling neutral about this.",
        "That's disgusting.",
        "You're being quite mischievous!",
        "This is a nightmare.",
        "I'm really angry about this."
    ]
    print("Quick test results:")
    for text in test_texts:
        emotion = classify_emotion(text)
        print(f"'{text}' -> {emotion}")