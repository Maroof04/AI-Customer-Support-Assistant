"""
Model Training and Knowledge Base Indexing Pipeline.
Trains the Intent Classifier on held-out train/test split, generates rigorous
evaluation metrics, indexes the Knowledge Base with FAISS, and persists artifacts to models/.
"""

import os
import sys
import json
import pandas as pd
from sklearn.model_selection import train_test_split

# Ensure project root is on sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.intent_classifier import IntentClassifier
from src.retriever import FaissRetriever
from src.utils import get_logger, plot_confusion_matrix, plot_metrics_bar


def main():
    logger = get_logger("TrainPipeline")
    logger.info("Starting AI Customer Support Assistant Training Pipeline...")

    data_dir = os.path.join(PROJECT_ROOT, "data")
    models_dir = os.path.join(PROJECT_ROOT, "models")
    os.makedirs(models_dir, exist_ok=True)

    tickets_file = os.path.join(data_dir, "support_tickets.csv")
    kb_file = os.path.join(data_dir, "knowledge_base.csv")

    if not os.path.exists(tickets_file) or not os.path.exists(kb_file):
        logger.error("Dataset files missing in data/. Run build scripts first.")
        sys.exit(1)

    # 1. Load Data
    df_tickets = pd.read_csv(tickets_file)
    df_kb = pd.read_csv(kb_file)
    logger.info(f"Loaded {len(df_tickets)} support tickets and {len(df_kb)} FAQ articles.")

    # 2. Stratified Train / Test Split (80% Train, 20% Held-out Test)
    # Stratification guarantees uniform class representation across splits, preventing data leakage
    X = df_tickets["query"].tolist()
    y = df_tickets["intent"].tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )
    logger.info(f"Split dataset: {len(X_train)} training samples, {len(X_test)} held-out test samples.")

    # 3. Train Intent Classifier
    logger.info("Training TF-IDF + Calibrated Logistic Regression Intent Classifier...")
    classifier = IntentClassifier()
    classifier.train(X_train, y_train, model_type="logistic_regression")

    # 4. Evaluate on Held-out Test Split
    logger.info("Evaluating classifier performance on held-out test set...")
    metrics = classifier.evaluate(X_test, y_test)

    print("\n" + "=" * 60)
    print("      HELD-OUT TEST SET EVALUATION RESULTS")
    print("=" * 60)
    print(f"  Accuracy:           {metrics['accuracy']:.2%}")
    print(f"  Weighted Precision: {metrics['precision_weighted']:.2%}")
    print(f"  Weighted Recall:    {metrics['recall_weighted']:.2%}")
    print(f"  Weighted F1-Score:  {metrics['f1_weighted']:.2%}")
    print(f"  Macro F1-Score:     {metrics['f1_macro']:.2%}")
    print("=" * 60)
    print("\nDetailed Per-Class Performance:")
    cr_df = pd.DataFrame(metrics["classification_report"]).transpose()
    print(cr_df[["precision", "recall", "f1-score", "support"]].to_string())
    print("=" * 60 + "\n")

    # Save Classifier Artifacts
    clf_path = os.path.join(models_dir, "intent_classifier.joblib")
    classifier.save(clf_path)
    logger.info(f"Saved trained intent classifier to: {clf_path}")

    # 5. Build FAISS Knowledge Base Index
    logger.info("Building FAISS vector index for knowledge base articles...")
    retriever = FaissRetriever()
    retriever.fit_from_dataframe(df_kb)
    retriever.save(models_dir)
    logger.info(f"Saved FAISS index ({retriever.dimension}D) and metadata to: {models_dir}")

    # 6. Save Evaluation Metrics and Plots
    metrics_json_path = os.path.join(models_dir, "evaluation_metrics.json")
    with open(metrics_json_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Saved metrics JSON to: {metrics_json_path}")

    cm_plot_path = os.path.join(models_dir, "confusion_matrix.png")
    plot_confusion_matrix(metrics["confusion_matrix"], metrics["classes"], save_path=cm_plot_path)

    bar_plot_path = os.path.join(models_dir, "metrics_summary.png")
    plot_metrics_bar(metrics, save_path=bar_plot_path)
    logger.info("Saved evaluation visualization charts.")

    logger.info("Training pipeline completed successfully!")


if __name__ == "__main__":
    main()
