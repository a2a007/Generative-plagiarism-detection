import os
import sys
import json
import pickle
import time
import argparse
from pathlib import Path

# Add src to python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.config import load_config
from src.utils.logging_utils import setup_logger
from src.classification.feature_extractor import FeatureExtractor
from src.classification.train_classifier import PlagiarismClassifierPipeline

logger = setup_logger("Classifier_Trainer")

def main():
    parser = argparse.ArgumentParser(description="Train and Evaluate Plagiarism Classifier and Export Pickle File")
    parser.add_argument("--candidates", type=str, default="outputs/candidate_fusion_results.jsonl", help="Path to candidate_fusion_results.jsonl")
    parser.add_argument("--output-pickle", type=str, default="outputs/trained_plagiarism_classifier.pkl", help="Output .pkl filepath")
    parser.add_argument("--model-type", type=str, default="random_forest", choices=["random_forest", "gradient_boosting"], help="Classifier architecture")
    args = parser.parse_args()

    cand_path = Path(args.candidates)
    if not cand_path.exists():
        raise FileNotFoundError(f"Candidate results file not found at: {cand_path}. Run scripts/run_experiment.py first.")

    logger.info(f"Loading candidate records from: {cand_path}")
    records = []
    with open(cand_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    logger.info(f"Extracting features for {len(records)} query candidate sets...")
    extractor = FeatureExtractor()
    X, y, feature_names = extractor.extract_features_from_candidates(records)

    logger.info(f"Extracted {X.shape[0]} candidate pairs (Features: {X.shape[1]}). Positive samples: {y.sum()}, Negative: {len(y) - y.sum()}")

    logger.info(f"Training and Evaluating {args.model_type} Classifier...")
    pipeline = PlagiarismClassifierPipeline(model_type=args.model_type)
    metrics, pkl_data = pipeline.train_and_evaluate(X, y, feature_names)

    pkl_path = Path(args.output_pickle)
    pkl_path.parent.mkdir(parents=True, exist_ok=True)
    with open(pkl_path, "wb") as f_pkl:
        pickle.dump(pkl_data, f_pkl)

    logger.info(f"\nSaved Trained Model Pickle File to: {pkl_path}")

    # Save JSON report
    json_path = Path("outputs/classification_report.json")
    with open(json_path, "w", encoding="utf-8") as f_json:
        json.dump(metrics, f_json, indent=2)

    logger.info(f"Saved Classification Metrics Report to: {json_path}")

    # Formatted terminal output
    logger.info("\n" + "=" * 60)
    logger.info("CLASSIFICATION PERFORMANCE METRICS REPORT")
    logger.info("=" * 60)
    logger.info(f"Model Architecture  : {args.model_type.upper()}")
    logger.info(f"Train Set Size      : {metrics['train_samples']} candidate pairs")
    logger.info(f"Test Set Size       : {metrics['test_samples']} candidate pairs")
    logger.info("-" * 60)
    logger.info(f"Accuracy            : {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    logger.info(f"Precision (Binary)  : {metrics['precision_binary']:.4f}")
    logger.info(f"Recall (Binary)     : {metrics['recall_binary']:.4f}")
    logger.info(f"F1 Score (Binary)   : {metrics['f1_binary']:.4f}")
    logger.info(f"F1 Score (Macro)    : {metrics['f1_macro']:.4f}")
    logger.info(f"F1 Score (Weighted) : {metrics['f1_weighted']:.4f}")
    logger.info(f"ROC-AUC Score       : {metrics['roc_auc']:.4f}")
    logger.info("-" * 60)
    cm = metrics["confusion_matrix"]
    logger.info(f"Confusion Matrix    : TP={cm['true_positive']}, FP={cm['false_positive']}, TN={cm['true_negative']}, FN={cm['false_negative']}")
    logger.info("-" * 60)
    logger.info("Feature Importances:")
    for feat, imp in metrics["feature_importances"].items():
        logger.info(f"  - {feat:<25}: {imp:.4f}")
    logger.info("=" * 60)

if __name__ == "__main__":
    main()
