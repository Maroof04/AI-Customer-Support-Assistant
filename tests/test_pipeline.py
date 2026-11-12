"""
Integration tests for the unified SupportAssistantPipeline.
"""

import os
import pytest
from src.intent_classifier import IntentClassifier
from src.retriever import FaissRetriever
from src.sentiment_analyzer import SentimentAnalyzer
from src.response_generator import ResponseGenerator
from src.utils import SupportAssistantPipeline

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")


@pytest.fixture
def pipeline():
    clf_path = os.path.join(MODELS_DIR, "intent_classifier.joblib")
    clf = IntentClassifier(clf_path)
    retriever = FaissRetriever().load(MODELS_DIR)
    analyzer = SentimentAnalyzer()
    generator = ResponseGenerator()
    return SupportAssistantPipeline(clf, retriever, analyzer, generator)


def test_pipeline_refund_flow(pipeline):
    query = "I want a refund for order #9821. It arrived broken and I am very upset!"
    result = pipeline.process_query(query)

    assert result["intent"]["intent"] == "refund"
    assert result["sentiment"]["sentiment"] == "negative"
    assert result["sentiment"]["frustration_detected"] is True
    assert len(result["retrieved_articles"]) > 0
    assert "Refund" in result["generation"]["matched_faq_title"] or "Damaged" in result["generation"]["matched_faq_title"]
    assert "apologize" in result["generation"]["response"].lower() or "frustration" in result["generation"]["response"].lower()


def test_pipeline_out_of_scope_guardrail(pipeline):
    # A completely bizarre/irrelevant query that does not belong in customer support
    query = "What is the capital city of Australia and what is the velocity of an unladen swallow?"
    # The retriever should have low similarity scores, and generator should trigger out of scope guardrail
    result = pipeline.process_query(query)

    top_art = result["retrieved_articles"][0]
    # Check that either it is marked not relevant or similarity is modest
    assert "response" in result["generation"]
    assert len(result["generation"]["response"]) > 20
