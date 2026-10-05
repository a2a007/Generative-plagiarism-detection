import os
import pickle
import numpy as np
from typing import Dict, Any, Tuple
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

class PlagiarismClassifierPipeline:
    """
    Train, evaluate, and pickle a machine learning model for plagiarism attribution classification.
    """
    def __init__(self, model_type: str = "random_forest", random_state: int = 42):
        self.model_type = model_type
        self.random_state = random_state
        self.scaler = StandardScaler()
        
        if model_type == "gradient_boosting":
            self.model = GradientBoostingClassifier(n_estimators=100, random_state=random_state)
        else:
            self.model = RandomForestClassifier(n_estimators=100, random_state=random_state, class_weight="balanced")

    def train_and_evaluate(self, X: np.ndarray, y: np.ndarray, feature_names: list, test_size: float = 0.2) -> Tuple[Dict[str, Any], Any]:
        """
        Splits data, scales features, trains model, and calculates full metrics suite.
        """
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state, stratify=y
        )

        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        self.model.fit(X_train_scaled, y_train)

        y_pred = self.model.predict(X_test_scaled)
        y_prob = self.model.predict_proba(X_test_scaled)[:, 1] if hasattr(self.model, "predict_proba") else y_pred

        acc = accuracy_score(y_test, y_pred)
        prec_binary = precision_score(y_test, y_pred, pos_label=1, zero_division=0)
        rec_binary = recall_score(y_test, y_pred, pos_label=1, zero_division=0)
        f1_binary = f1_score(y_test, y_pred, pos_label=1, zero_division=0)

        prec_macro = precision_score(y_test, y_pred, average="macro", zero_division=0)
        rec_macro = recall_score(y_test, y_pred, average="macro", zero_division=0)
        f1_macro = f1_score(y_test, y_pred, average="macro", zero_division=0)
        f1_weighted = f1_score(y_test, y_pred, average="weighted", zero_division=0)

        try:
            auc = roc_auc_score(y_test, y_prob)
        except Exception:
            auc = 0.0

        cm = confusion_matrix(y_test, y_pred).tolist()

        feature_importances = {}
        if hasattr(self.model, "feature_importances_"):
            for name, imp in zip(feature_names, self.model.feature_importances_):
                feature_importances[name] = float(imp)

        metrics = {
            "accuracy": float(acc),
            "precision_binary": float(prec_binary),
            "recall_binary": float(rec_binary),
            "f1_binary": float(f1_binary),
            "precision_macro": float(prec_macro),
            "recall_macro": float(rec_macro),
            "f1_macro": float(f1_macro),
            "f1_weighted": float(f1_weighted),
            "roc_auc": float(auc),
            "confusion_matrix": {
                "true_negative": cm[0][0] if len(cm) > 0 else 0,
                "false_positive": cm[0][1] if len(cm) > 0 else 0,
                "false_negative": cm[1][0] if len(cm) > 1 else 0,
                "true_positive": cm[1][1] if len(cm) > 1 else 0
            },
            "train_samples": len(X_train),
            "test_samples": len(X_test),
            "feature_importances": feature_importances
        }

        return metrics, {
            "model": self.model,
            "scaler": self.scaler,
            "feature_names": feature_names,
            "metrics": metrics
        }
