"""
Configuration settings for EmoTiny emotion classification.
"""

EMOTION_LABELS = [
    "neutral",
    "happy", 
    "sad",
    "angry",
    "surprised",
    "disgusted",
    "mischievous",
    "love",
    "nightmare"
]

EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
EMBEDDING_DIM = 384  # Dimension of the chosen model

DEFAULT_TRAIN_CONFIG = {
    "test_size": 0.2,
    "random_state": 42,
    "classifier_type": "mlp",  # "logistic" or "mlp"
    "mlp_hidden_sizes": (128, 64),
    "mlp_activation": "relu",
    "mlp_max_iter": 1000,
    "mlp_early_stopping": True,
    "mlp_validation_fraction": 0.1,
    "logistic_max_iter": 1000,
    "logistic_solver": "lbfgs",
    "cross_validation_folds": 5
}

QUANTIZATION_CONFIG = {
    "onnx_opset_version": 14,
    "optimize_for_mobile": False,
    "quantization_mode": "dynamic"  # "dynamic" or "static"
}

INFERENCE_CONFIG = {
    "batch_size": 1,
    "use_onnx": True,  # Use ONNX runtime for faster inference
    "device": "cpu"    # "cpu" or "cuda"
}