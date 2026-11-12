"""
Utility Functions and Unified Pipeline Orchestrator.
Connects Preprocessing, Intent Classification, Sentiment Analysis,
FAISS Retrieval, and Response Generation into a single source of truth.
"""

import os
import logging
from typing import Dict, Any, Optional, List
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from src.preprocessing import clean_text, extract_metadata
from src.intent_classifier import IntentClassifier
from src.sentiment_analyzer import SentimentAnalyzer
from src.retriever import FaissRetriever
from src.response_generator import ResponseGenerator


def get_logger(name: str = "SupportAssistant") -> logging.Logger:
    """Configures and returns a standardized console logger."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "[%(asctime)s] [%(levelname)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


class SupportAssistantPipeline:
    """
    End-to-End Customer Support Pipeline.
    Single point of entry that coordinates:
    Customer Query -> Preprocessing -> Intent Classifier + Sentiment Analyzer -> FAISS Retriever -> Response Generator
    """

    def __init__(
        self,
        classifier: IntentClassifier,
        retriever: FaissRetriever,
        sentiment_analyzer: Optional[SentimentAnalyzer] = None,
        generator: Optional[ResponseGenerator] = None
    ):
        self.classifier = classifier
        self.retriever = retriever
        self.sentiment_analyzer = sentiment_analyzer or SentimentAnalyzer()
        self.generator = generator or ResponseGenerator()
        self.logger = get_logger("Pipeline")

    def process_query(self, query: str, top_k: int = 2) -> Dict[str, Any]:
        """
        Executes the full pipeline for an incoming customer query.
        Returns a rich structured dictionary containing all intermediate and final outputs.
        """
        raw_query = query.strip()
        cleaned_query = clean_text(raw_query)
        extracted_entities = extract_metadata(raw_query)

        # 1. Intent Classification
        intent_info = self.classifier.predict(raw_query)

        # 2. Sentiment Analysis & Tone Detection
        sentiment_info = self.sentiment_analyzer.analyze(raw_query)

        # 3. Knowledge Base Retrieval (with category hinting)
        detected_category = intent_info.get("intent")
        filter_cat = detected_category if detected_category != "general_inquiry" else None
        retrieved_articles = self.retriever.retrieve(
            query=raw_query,
            top_k=top_k,
            threshold=0.20,
            category_filter=filter_cat
        )

        # 4. Context-Aware Grounded Response Generation
        generation_result = self.generator.generate(
            query=raw_query,
            intent_info=intent_info,
            sentiment_info=sentiment_info,
            retrieved_articles=retrieved_articles
        )

        return {
            "query": raw_query,
            "cleaned_query": cleaned_query,
            "entities": extracted_entities,
            "intent": intent_info,
            "sentiment": sentiment_info,
            "retrieved_articles": retrieved_articles,
            "generation": generation_result
        }


def plot_confusion_matrix(cm: List[List[int]], classes: List[str], save_path: Optional[str] = None):
    """Generates and optionally saves a heatmap of the classification confusion matrix."""
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=classes,
        yticklabels=classes,
        ax=ax
    )
    ax.set_title("Intent Classification Confusion Matrix", fontsize=14, pad=12)
    ax.set_xlabel("Predicted Intent", fontsize=12)
    ax.set_ylabel("True Intent", fontsize=12)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
    return fig


def plot_metrics_bar(metrics: Dict[str, float], save_path: Optional[str] = None):
    """Plots key evaluation metrics (Accuracy, Precision, Recall, F1)."""
    fig, ax = plt.subplots(figsize=(7, 4))
    keys = ["accuracy", "precision_weighted", "recall_weighted", "f1_weighted"]
    labels = ["Accuracy", "Weighted Precision", "Weighted Recall", "Weighted F1"]
    values = [metrics.get(k, 0.0) for k in keys]

    bars = ax.bar(labels, values, color=["#2563eb", "#3b82f6", "#60a5fa", "#93c5fd"])
    ax.set_ylim(0, 1.1)
    ax.set_title("Intent Classifier Evaluation Metrics", fontsize=13)
    ax.set_ylabel("Score (0.0 - 1.0)", fontsize=11)

    for bar in bars:
        h = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.,
            h + 0.02,
            f"{h:.1%}",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold"
        )

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
    return fig
