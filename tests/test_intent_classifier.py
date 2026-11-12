"""
Unit tests for Intent Classifier training, prediction, and persistence.
"""

import os
import pytest
from src.intent_classifier import IntentClassifier


@pytest.fixture
def sample_dataset():
    texts = [
        "Where is my refund for the returned sneakers?",
        "Double charged on my Visa card, please refund the second charge.",
        "My credit card was declined during checkout.",
        "Can I pay with Apple Pay or PayPal?",
        "Where is my delivery tracking number? Package is late.",
        "Parcel arrived broken and damaged by courier.",
        "Cancel order #9912 immediately before dispatch.",
        "Stop auto-renewal on my monthly subscription plan.",
        "Forgot my password and need account recovery reset link.",
        "Account is locked after too many incorrect password tries.",
        "Mobile app keeps crashing every time I open the cart.",
        "Website throws HTTP 500 error on checkout page."
    ]
    labels = [
        "refund", "refund",
        "payment", "payment",
        "delivery", "delivery",
        "cancellation", "cancellation",
        "account_issue", "account_issue",
        "technical_support", "technical_support"
    ]
    return texts, labels


def test_train_and_predict(sample_dataset):
    texts, labels = sample_dataset
    clf = IntentClassifier()
    clf.train(texts, labels, model_type="logistic_regression")

    # In a small 12-sample fixture, check raw_intent and probability distribution
    res = clf.predict("I want to return an item and get my refund", threshold=0.15)
    assert res["raw_intent"] in clf.classes_
    assert res["intent"] in clf.classes_
    assert "probabilities" in res
    assert 0.0 <= res["confidence"] <= 1.0


def test_confidence_threshold_fallback(sample_dataset):
    texts, labels = sample_dataset
    clf = IntentClassifier()
    clf.train(texts, labels)

    # Empty or gibberish input should be marked as general_inquiry or low confidence
    res = clf.predict("xyz abc 1234567890", threshold=0.99)
    assert res["is_confident"] is False
    assert res["intent"] == "general_inquiry"


def test_save_and_load(tmp_path, sample_dataset):
    texts, labels = sample_dataset
    clf = IntentClassifier()
    clf.train(texts, labels)

    model_file = str(tmp_path / "model.joblib")
    clf.save(model_file)
    assert os.path.exists(model_file)

    loaded = IntentClassifier(model_file)
    assert loaded.is_trained is True
    res = loaded.predict("Can I get a refund?")
    assert res["raw_intent"] == "refund"
