"""
Unit tests for Sentiment Analyzer and Tone Adaptation.
"""

import pytest
from src.sentiment_analyzer import SentimentAnalyzer


@pytest.fixture
def analyzer():
    return SentimentAnalyzer()


def test_positive_sentiment(analyzer):
    text = "Thank you so much! The support team was fantastic and fast."
    res = analyzer.analyze(text)
    assert res["sentiment"] == "positive"
    assert res["compound_score"] > 0.05
    assert res["tone_recommendation"]["style"] == "Warm & Appreciative"


def test_negative_sentiment_and_frustration(analyzer):
    text = "This is unacceptable! You guys charged me twice and your service is horrible!!"
    res = analyzer.analyze(text)
    assert res["sentiment"] == "negative"
    assert res["frustration_detected"] is True
    assert "Empathetic" in res["tone_recommendation"]["style"]


def test_neutral_sentiment(analyzer):
    text = "What is the return address for orders shipped to California?"
    res = analyzer.analyze(text)
    assert res["sentiment"] == "neutral"
    assert "Direct" in res["tone_recommendation"]["style"]


def test_empty_string_handling(analyzer):
    res = analyzer.analyze("")
    assert res["sentiment"] == "neutral"
    assert res["compound_score"] == 0.0
