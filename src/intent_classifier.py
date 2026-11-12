"""
Intent Classifier Module for Customer Support Assistant.
Utilizes TF-IDF feature extraction with calibrated Machine Learning models
(Logistic Regression, Multinomial Naive Bayes, Linear SVC with probability calibration).
"""

import os
import joblib
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix
)
from src.preprocessing import clean_text


SUPPORTED_INTENTS = [
    "refund",
    "payment",
    "delivery",
    "cancellation",
    "account_issue",
    "technical_support"
]

CONFIDENCE_THRESHOLD = 0.35


class IntentClassifier:
    """
    Modular Intent Classifier supporting TF-IDF vectorization and classical ML models.
    Provides calibrated probability estimates and confidence thresholding.
    """

    def __init__(self, model_path: Optional[str] = None):
        self.pipeline: Optional[Pipeline] = None
        self.classes_: Optional[List[str]] = None
        self.is_trained: bool = False

        if model_path and os.path.exists(model_path):
            self.load(model_path)

    def _build_pipeline(self, model_type: str = "logistic_regression") -> Pipeline:
        """
        Builds the Scikit-learn Pipeline with optimized TF-IDF vectorizer parameters.
        - ngram_range=(1, 2): captures both single words ('refund') and phrases ('double charge')
        - sublinear_tf=True: applies sublinear scaling (1 + log(tf)) to reduce bias from long queries
        - min_df=1: accommodates small-to-medium specialized domain vocabularies
        """
        vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            sublinear_tf=True,
            min_df=1,
            strip_accents="unicode"
        )

        if model_type == "naive_bayes":
            clf = MultinomialNB(alpha=0.1)
        elif model_type == "logistic_regression":
            # Balanced class weights and calibrated regularization for generalizable decision boundaries
            clf = LogisticRegression(
                C=3.0,
                max_iter=1000,
                class_weight="balanced",
                random_state=42
            )
        else:
            clf = LogisticRegression(C=1.0, max_iter=1000, random_state=42)

        return Pipeline([
            ("tfidf", vectorizer),
            ("classifier", clf)
        ])

    def train(self, texts: List[str], labels: List[str], model_type: str = "logistic_regression") -> "IntentClassifier":
        """
        Trains the TF-IDF and Classifier pipeline on preprocessed training texts.
        """
        cleaned_texts = [clean_text(t) for t in texts]
        self.pipeline = self._build_pipeline(model_type=model_type)
        self.pipeline.fit(cleaned_texts, labels)
        self.classes_ = list(self.pipeline.classes_)
        self.is_trained = True
        return self

    def evaluate(self, texts: List[str], labels: List[str]) -> Dict[str, Any]:
        """
        Evaluates classifier performance on a held-out test split.
        Returns accuracy, precision, recall, f1-score, and detailed classification report.
        """
        if not self.is_trained or self.pipeline is None:
            raise ValueError("Classifier must be trained before evaluation.")

        cleaned_texts = [clean_text(t) for t in texts]
        predictions = self.pipeline.predict(cleaned_texts)

        acc = float(accuracy_score(labels, predictions))
        prec, rec, f1, _ = precision_recall_fscore_support(
            labels, predictions, average="weighted", zero_division=0
        )
        prec_macro, rec_macro, f1_macro, _ = precision_recall_fscore_support(
            labels, predictions, average="macro", zero_division=0
        )

        rep_dict = classification_report(
            labels, predictions, output_dict=True, zero_division=0
        )
        cm = confusion_matrix(labels, predictions, labels=self.classes_)

        return {
            "accuracy": round(acc, 4),
            "precision_weighted": round(float(prec), 4),
            "recall_weighted": round(float(rec), 4),
            "f1_weighted": round(float(f1), 4),
            "precision_macro": round(float(prec_macro), 4),
            "recall_macro": round(float(rec_macro), 4),
            "f1_macro": round(float(f1_macro), 4),
            "classification_report": rep_dict,
            "confusion_matrix": cm.tolist(),
            "classes": self.classes_
        }

    def predict(self, text: str, threshold: float = CONFIDENCE_THRESHOLD) -> Dict[str, Any]:
        """
        Predicts the intent for a customer query along with probability distribution.
        If the highest probability is below threshold, flags query as unconfident.
        """
        if not self.is_trained or self.pipeline is None:
            raise ValueError("Classifier has not been trained or loaded.")

        cleaned = clean_text(text)
        if not cleaned:
            return {
                "intent": "general_inquiry",
                "confidence": 0.0,
                "probabilities": {cls: 0.0 for cls in self.classes_ or SUPPORTED_INTENTS},
                "is_confident": False
            }

        probabilities = self.pipeline.predict_proba([cleaned])[0]
        max_idx = int(np.argmax(probabilities))
        confidence = float(probabilities[max_idx])
        top_intent = self.classes_[max_idx]

        prob_dict = {
            cls: round(float(prob), 4)
            for cls, prob in zip(self.classes_, probabilities)
        }
        # Sort probability dictionary descending
        prob_dict = dict(sorted(prob_dict.items(), key=lambda item: item[1], reverse=True))

        is_confident = confidence >= threshold
        final_intent = top_intent if is_confident else "general_inquiry"

        return {
            "intent": final_intent,
            "raw_intent": top_intent,
            "confidence": round(confidence, 4),
            "probabilities": prob_dict,
            "is_confident": is_confident
        }

    def save(self, filepath: str) -> None:
        """Serializes the trained pipeline and classes metadata to disk using joblib."""
        if not self.is_trained or self.pipeline is None:
            raise ValueError("Cannot save an untrained classifier.")
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        payload = {
            "pipeline": self.pipeline,
            "classes": self.classes_
        }
        joblib.dump(payload, filepath)

    def load(self, filepath: str) -> "IntentClassifier":
        """Loads a pre-trained pipeline and class labels from disk."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found at: {filepath}")
        payload = joblib.load(filepath)
        self.pipeline = payload["pipeline"]
        self.classes_ = payload["classes"]
        self.is_trained = True
        return self
