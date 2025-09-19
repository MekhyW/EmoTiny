"""
EmoTiny: Fast multilingual emotion classification for real-time applications.

A lightweight emotion classifier designed for real-time animated face control,
optimized for low latency and small footprint deployment.
"""

__author__ = "Mekhy W.!"

from .inference import classify_emotion, EmoTinyClassifier
from .training import EmoTinyTrainer
from .preprocessing import EmoTinyPreprocessor

__all__ = [
    "classify_emotion",
    "EmoTinyClassifier", 
    "EmoTinyTrainer",
    "EmoTinyPreprocessor"
]