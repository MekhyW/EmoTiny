"""
EmoTiny: Fast multilingual emotion classification for real-time applications.

A lightweight emotion classifier designed for real-time animated face control,
optimized for low latency and small footprint deployment.
"""

__author__ = "Mekhy W.!"

from .inference import load_model, classify_emotion, classify_emotion_with_confidence, EmoTinyClassifier
from .training import EmoTinyTrainer
from .preprocessing import EmoTinyPreprocessor

__all__ = [
    "load_model",
    "classify_emotion",
    "classify_emotion_with_confidence",
    "EmoTinyClassifier", 
    "EmoTinyTrainer",
    "EmoTinyPreprocessor"
]