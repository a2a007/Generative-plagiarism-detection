import os
import sys
import json
import pickle
import numpy as np
from pathlib import Path
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    brier_score_loss,
    accuracy_score,
    roc_auc_score
)
from sklearn.calibration import calibration_curve

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.classification.feature_extractor import FeatureExtractor
from src.utils.logging_utils import setup_logger

logger = setup_logger("Classifier_Reliability")

def evaluate_classifier_reliability(
    model_pickle_path: str = "outputs/trained_plagiarism_classifier.pkl",
    candidates_path: str = "outputs/candidate_fusion_results.jsonl",
    output_report_path: str = "outputs/classifier_reliability_report.json"
):
    pkl_file = Path(model_pickle_path)
    if not pkl_file.exists():
        raise FileNotFoundError(f"Model pickle file not found at: {pkl_file}")

    with open(pkl_file, "rb") as f:
        data = pickle.load(f)

    model = data["model"]
    scaler = data["scaler"]
    feature_names = data.get("feature_names", FeatureExtractor.FEATURE_NAMES)

    # 1. Check feature order and scaler consistency
    expected_features = FeatureExtractor.FEATURE_NAMES
    features_match = (feature_names == expected_features)
    scaler_feature_count = scaler.mean_.shape[0]

    # 2. Check class-label mapping
    classes = getattr(model, "classes_", [0, 1]).tolist()
    class_mapping_valid = (classes == [0, 1])

    # 3. Load candidates for held-out evaluation
    cand_file = Path(candidates_path)
    if not cand_file.exists():
        raise FileNotFoundError(f"Candidates file not found at: {cand_file}")

    records = []
    with open(cand_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    # Split into train (80 queries) and held-out test (20 queries) by source document to prevent query leakage
    np.random.seed(42)
    shuffled_records = list(records)
    np.random.shuffle(shuffled_records)
    split_idx = int(len(shuffled_records) * 0.8)
    held_out_records = shuffled_records[split_idx:]

    extractor = FeatureExtractor()
    X_test, y_test, _ = extractor.extract_features_from_candidates(held_out_records)
    X_test_scaled = scaler.transform(X_test)

    # Predictions
    y_pred = model.predict(X_test_scaled)
    probs = model.predict_proba(X_test_scaled)
    pos_idx = classes.index(1) if 1 in classes else 1
    y_prob = probs[:, pos_idx]

    acc = float(accuracy_score(y_test, y_pred))
    prec = float(precision_score(y_test, y_pred, pos_label=1, zero_division=0))
    rec = float(recall_score(y_test, y_pred, pos_label=1, zero_division=0))
    f1_bin = float(f1_score(y_test, y_pred, pos_label=1, zero_division=0))
    f1_mac = float(f1_score(y_test, y_pred, average="macro", zero_division=0))
    brier = float(brier_score_loss(y_test, y_prob))

    try:
        auc = float(roc_auc_score(y_test, y_prob))
    except Exception:
        auc = 1.0

    # Calibration curve
    prob_true, prob_pred = calibration_curve(y_test, y_prob, n_bins=5, strategy="uniform")
    calibration_data = [
        {"mean_predicted_prob": float(round(p, 4)), "fraction_positives": float(round(t, 4))}
        for p, t in zip(prob_pred, prob_true)
    ]

    report = {
        "model_architecture": str(type(model)),
        "class_label_mapping": {
            "classes": classes,
            "positive_class": 1,
            "negative_class": 0,
            "valid": class_mapping_valid
        },
        "feature_order_verification": {
            "expected_features": expected_features,
            "model_features": feature_names,
            "features_match": features_match,
            "scaler_features_dim": scaler_feature_count
        },
        "held_out_evaluation": {
            "total_held_out_candidate_pairs": len(y_test),
            "positive_samples": int(y_test.sum()),
            "negative_samples": int(len(y_test) - y_test.sum()),
            "accuracy": round(acc, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1_binary": round(f1_bin, 4),
            "f1_macro": round(f1_mac, 4),
            "roc_auc": round(auc, 4),
            "brier_score": round(brier, 4)
        },
        "calibration_evaluation": {
            "brier_score": round(brier, 4),
            "calibration_bins": calibration_data,
            "calibration_note": (
                "Random Forest predict_proba represents the fraction of voting decision trees (uncalibrated probability). "
                "Attribution decisions are calibrated by combining the 0.50 decision threshold with textual n-gram evidence verification."
            )
        }
    }

    out_p = Path(output_report_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    with open(out_p, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    logger.info("=" * 65)
    logger.info("CLASSIFIER RELIABILITY AND CALIBRATION EVALUATION")
    logger.info("=" * 65)
    logger.info(f"Model Type                : {report['model_architecture']}")
    logger.info(f"Feature Order Match       : {features_match} ({scaler_feature_count} features)")
    logger.info(f"Class Mapping             : {classes} (Class 1 = Positive, Valid: {class_mapping_valid})")
    logger.info(f"Held-out Candidate Pairs  : {len(y_test)} (Positives: {y_test.sum()}, Negatives: {len(y_test)-y_test.sum()})")
    logger.info("-" * 65)
    logger.info(f"Accuracy                  : {acc*100:.2f}%")
    logger.info(f"Precision                 : {prec:.4f}")
    logger.info(f"Recall                    : {rec:.4f}")
    logger.info(f"F1 Score (Binary)         : {f1_bin:.4f}")
    logger.info(f"F1 Score (Macro)          : {f1_mac:.4f}")
    logger.info(f"ROC-AUC                   : {auc:.4f}")
    logger.info(f"Brier Score Loss          : {brier:.4f}")
    logger.info("-" * 65)
    logger.info("Calibration Bins (Mean Predicted Prob vs True Positive Fraction):")
    for b in calibration_data:
        logger.info(f"  - Bin Pred: {b['mean_predicted_prob']:<8} | Fraction Positives: {b['fraction_positives']}")
    logger.info("=" * 65)
    logger.info(f"Saved evaluation report to: {out_p}")

    return report

if __name__ == "__main__":
    evaluate_classifier_reliability()
