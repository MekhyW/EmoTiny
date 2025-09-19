"""
Training pipeline for EmoTiny emotion classification.
"""

import os
import joblib
import numpy as np
from typing import Dict, List, Optional, Any
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from .preprocessing import EmoTinyPreprocessor
from .config import DEFAULT_TRAIN_CONFIG, EMOTION_LABELS


class EmoTinyTrainer:
    """
    Trainer class for emotion classification models.
    Supports both Logistic Regression and MLP classifiers.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the trainer.
        
        Args:
            config: Training configuration dictionary
        """
        self.config = {**DEFAULT_TRAIN_CONFIG, **(config or {})}
        self.preprocessor = EmoTinyPreprocessor()
        self.classifier = None
        self.training_history = {}
        self.label_to_idx = {label: idx for idx, label in enumerate(EMOTION_LABELS)}
        self.idx_to_label = {idx: label for idx, label in enumerate(EMOTION_LABELS)}
        
    def _create_classifier(self) -> Any:
        """Create classifier based on configuration."""
        if self.config["classifier_type"] == "logistic":
            return LogisticRegression(
                max_iter=self.config["logistic_max_iter"],
                solver=self.config["logistic_solver"],
                random_state=self.config["random_state"],
                multi_class="ovr",  # One-vs-Rest for multiclass
                class_weight="balanced"  # Handle class imbalance
            )
        elif self.config["classifier_type"] == "mlp":
            return MLPClassifier(
                hidden_layer_sizes=self.config["mlp_hidden_sizes"],
                activation=self.config["mlp_activation"],
                max_iter=self.config["mlp_max_iter"],
                early_stopping=self.config["mlp_early_stopping"],
                validation_fraction=self.config["mlp_validation_fraction"],
                random_state=self.config["random_state"],
                learning_rate="adaptive",
                alpha=0.001  # L2 regularization
            )
        else:
            raise ValueError(f"Unknown classifier type: {self.config['classifier_type']}")
    
    def train(self, texts: List[str], labels: List[str], save_path: Optional[str] = None, perform_cv: bool = True) -> Dict[str, Any]:
        """
        Train the emotion classifier.
        
        Args:
            texts: List of text samples
            labels: List of emotion labels
            save_path: Path to save the trained model
            perform_cv: Whether to perform cross-validation
            
        Returns:
            Dictionary with training results
        """
        print(f"Training EmoTiny classifier ({self.config['classifier_type']})...")
        print(f"Dataset size: {len(texts)} samples")
        labels = self.preprocessor.validate_labels(labels)
        X, y, X_val, y_val = self.preprocessor.prepare_training_data(texts, labels, validation_split=0.0)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.config["test_size"], random_state=self.config["random_state"], stratify=y)
        print(f"Training set: {len(X_train)} samples")
        print(f"Test set: {len(X_test)} samples")
        self.classifier = self._create_classifier()
        print("Training classifier...")
        self.classifier.fit(X_train, y_train)
        y_pred = self.classifier.predict(X_test)
        test_accuracy = accuracy_score(y_test, y_pred)
        print(f"Test accuracy: {test_accuracy:.4f}")
        cv_scores = None
        if perform_cv:
            print("Performing cross-validation...")
            cv_scores = cross_val_score(self.classifier, X_train, y_train, cv=self.config["cross_validation_folds"], scoring="accuracy")
            print(f"CV accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        report = classification_report(y_test, y_pred, target_names=EMOTION_LABELS, output_dict=True)
        self.training_history = {
            "test_accuracy": test_accuracy,
            "cv_scores": cv_scores,
            "classification_report": report,
            "confusion_matrix": confusion_matrix(y_test, y_pred),
            "config": self.config.copy()
        }
        if save_path:
            self.save_model(save_path)
        print("\nTraining completed!")
        print(f"Final test accuracy: {test_accuracy:.4f}")
        if cv_scores is not None:
            print(f"Cross-validation accuracy: {cv_scores.mean():.4f}")
        return self.training_history
    
    def hyperparameter_search(self, texts: List[str], labels: List[str], param_grid: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Perform hyperparameter search using GridSearchCV.
        
        Args:
            texts: List of text samples
            labels: List of emotion labels
            param_grid: Parameter grid for search
            
        Returns:
            Best parameters and scores
        """
        print("Performing hyperparameter search...")
        if param_grid is None:
            if self.config["classifier_type"] == "logistic":
                param_grid = {
                    "C": [0.1, 1.0, 10.0],
                    "solver": ["lbfgs", "liblinear"],
                    "max_iter": [500, 1000]
                }
            else:  # MLP
                param_grid = {
                    "hidden_layer_sizes": [(64,), (128,), (128, 64), (256, 128)],
                    "alpha": [0.0001, 0.001, 0.01],
                    "learning_rate": ["constant", "adaptive"]
                }
        labels = self.preprocessor.validate_labels(labels)
        X, y, _, _ = self.preprocessor.prepare_training_data(texts, labels)
        base_classifier = self._create_classifier()
        grid_search = GridSearchCV(base_classifier, param_grid, cv=3, scoring="accuracy", n_jobs=-1, verbose=1)
        grid_search.fit(X, y)
        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Best CV score: {grid_search.best_score_:.4f}")
        self.config.update(grid_search.best_params_)
        self.classifier = grid_search.best_estimator_
        return {
            "best_params": grid_search.best_params_,
            "best_score": grid_search.best_score_,
            "cv_results": grid_search.cv_results_
        }
    
    def evaluate_model(self, texts: List[str], labels: List[str]) -> Dict[str, Any]:
        """
        Evaluate the trained model on new data.
        
        Args:
            texts: List of text samples
            labels: List of true emotion labels
            
        Returns:
            Evaluation metrics
        """
        if self.classifier is None:
            raise ValueError("Model not trained yet. Call train() first.")
        labels = self.preprocessor.validate_labels(labels)
        X = self.preprocessor.encode_texts(texts)
        y_true = np.array([self.label_to_idx[label] for label in labels])
        y_pred = self.classifier.predict(X)
        accuracy = accuracy_score(y_true, y_pred)
        report = classification_report(y_true, y_pred, target_names=EMOTION_LABELS, output_dict=True)
        cm = confusion_matrix(y_true, y_pred)
        return {
            "accuracy": accuracy,
            "classification_report": report,
            "confusion_matrix": cm
        }
    
    def plot_confusion_matrix(self, save_path: Optional[str] = None):
        """Plot confusion matrix from training history."""
        if "confusion_matrix" not in self.training_history:
            raise ValueError("No confusion matrix available. Train the model first.")
        cm = self.training_history["confusion_matrix"]
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=EMOTION_LABELS, yticklabels=EMOTION_LABELS)
        plt.title("Emotion Classification Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Confusion matrix saved to {save_path}")
        plt.show()
    
    def save_model(self, save_path: str):
        """
        Save the trained model and preprocessor.
        
        Args:
            save_path: Directory path to save the model
        """
        if self.classifier is None:
            raise ValueError("No trained model to save.")
        os.makedirs(save_path, exist_ok=True)
        classifier_path = os.path.join(save_path, "classifier.joblib")
        joblib.dump(self.classifier, classifier_path)
        metadata = {
            "config": self.config,
            "emotion_labels": EMOTION_LABELS,
            "label_to_idx": self.label_to_idx,
            "idx_to_label": self.idx_to_label,
            "embedding_model": self.preprocessor.model_name,
            "training_history": self.training_history
        }
        metadata_path = os.path.join(save_path, "metadata.joblib")
        joblib.dump(metadata, metadata_path)
        print(f"Model saved to {save_path}")
        print(f"Files: classifier.joblib, metadata.joblib")
    
    def load_model(self, model_path: str):
        """
        Load a trained model.
        
        Args:
            model_path: Directory path containing the saved model
        """
        classifier_path = os.path.join(model_path, "classifier.joblib")
        metadata_path = os.path.join(model_path, "metadata.joblib")
        if not os.path.exists(classifier_path) or not os.path.exists(metadata_path):
            raise FileNotFoundError(f"Model files not found in {model_path}")
        self.classifier = joblib.load(classifier_path)
        metadata = joblib.load(metadata_path)
        self.config = metadata["config"]
        self.training_history = metadata.get("training_history", {})
        if metadata["embedding_model"] != self.preprocessor.model_name:
            self.preprocessor = EmoTinyPreprocessor(metadata["embedding_model"])
        print(f"Model loaded from {model_path}")
    
    def get_feature_importance(self) -> Optional[np.ndarray]:
        """Get feature importance for logistic regression models."""
        if self.classifier is None:
            return None
        if hasattr(self.classifier, "coef_"):
            return np.abs(self.classifier.coef_).mean(axis=0)
        return None