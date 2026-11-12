"""
Unit tests for FAISS Knowledge Base Retriever.
"""

import os
import pytest
import pandas as pd
from src.retriever import FaissRetriever


@pytest.fixture
def sample_kb():
    return pd.DataFrame([
        {
            "faq_id": "FAQ-TEST-01",
            "category": "refund",
            "title": "Refund Timeline",
            "question": "How long does a refund take to process and when will money arrive?",
            "keywords": "refund timeline, bank credit days, money back, return money",
            "answer": "Refunds take 3-5 business days after inspection.",
            "action_steps": "1. Check return status. 2. Contact support.",
            "escalation_contact": "billing@example.com"
        },
        {
            "faq_id": "FAQ-TEST-02",
            "category": "technical_support",
            "title": "App Crashing Fix",
            "question": "Why does the app crash on checkout?",
            "keywords": "app crash, freeze, bug",
            "answer": "Clear your browser cache and update to latest version.",
            "action_steps": "1. Clear cache. 2. Restart device.",
            "escalation_contact": "tech@example.com"
        }
    ])


def test_fit_and_retrieve(sample_kb):
    retriever = FaissRetriever()
    retriever.fit_from_dataframe(sample_kb)

    results = retriever.retrieve("When will my money back arrive?", top_k=1)
    assert len(results) == 1
    assert results[0]["faq_id"] == "FAQ-TEST-01"
    assert results[0]["similarity_score"] > 0.1


def test_save_and_load(tmp_path, sample_kb):
    retriever = FaissRetriever()
    retriever.fit_from_dataframe(sample_kb)

    save_dir = str(tmp_path / "models")
    retriever.save(save_dir)

    loaded = FaissRetriever().load(save_dir)
    results = loaded.retrieve("The application keeps crashing", top_k=1)
    assert len(results) == 1
    assert results[0]["faq_id"] == "FAQ-TEST-02"
