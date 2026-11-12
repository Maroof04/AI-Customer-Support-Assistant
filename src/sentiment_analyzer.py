"""
Sentiment Analysis and Tone Adaptation Module.
Analyzes customer sentiment using VADER NLP lexicon scoring,
detects emotional frustration indicators, and provides actionable tone modulation.
"""

import re
from typing import Dict, Any
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


class SentimentAnalyzer:
    """
    Performs sentiment analysis and emotional intensity scoring on customer text.
    Provides tone recommendations for grounded customer response generation.
    """

    def __init__(self):
        self.analyzer = SentimentIntensityAnalyzer()

        # Urgency and frustration keywords commonly signaling high customer distress
        self.frustration_signals = {
            "furious", "horrible", "terrible", "worst", "unacceptable", "scam",
            "ridiculous", "disgusted", "angry", "cheated", "lawsuit", "incompetent"
        }
        self.urgency_signals = {
            "immediately", "right now", "urgent", "asap", "emergency", "blocked",
            "cannot wait", "hurry", "expedite", "at once"
        }

    def analyze(self, text: str) -> Dict[str, Any]:
        """
        Analyzes customer query text and returns:
        - sentiment: 'positive', 'neutral', 'negative'
        - scores: compound, pos, neu, neg
        - frustration_detected: bool
        - urgency_detected: bool
        - tone_recommendation: dict with persona, prefix, and guidelines
        """
        if not isinstance(text, str) or not text.strip():
            return {
                "sentiment": "neutral",
                "compound_score": 0.0,
                "scores": {"pos": 0.0, "neu": 1.0, "neg": 0.0, "compound": 0.0},
                "frustration_detected": False,
                "urgency_detected": False,
                "tone_recommendation": self._get_tone_guidance("neutral", False, False)
            }

        scores = self.analyzer.polarity_scores(text)
        compound = scores["compound"]

        # Classification thresholds
        if compound >= 0.05:
            sentiment = "positive"
        elif compound <= -0.05:
            sentiment = "negative"
        else:
            sentiment = "neutral"

        # Heuristic detection of elevated distress
        lower_text = text.lower()
        words = set(re.findall(r"\b\w+\b", lower_text))
        has_frustration = bool(words & self.frustration_signals) or ("!!" in text) or (scores["neg"] >= 0.25)
        has_urgency = bool(words & self.urgency_signals)

        # Tone adaptation instructions for the response generator
        tone_recommendation = self._get_tone_guidance(sentiment, has_frustration, has_urgency)

        return {
            "sentiment": sentiment,
            "compound_score": round(compound, 4),
            "scores": {k: round(v, 4) for k, v in scores.items()},
            "frustration_detected": has_frustration,
            "urgency_detected": has_urgency,
            "tone_recommendation": tone_recommendation
        }

    def _get_tone_guidance(self, sentiment: str, frustration: bool, urgency: bool) -> Dict[str, str]:
        """
        Maps customer emotional state to tailored conversational response tone.
        """
        if sentiment == "negative" or frustration:
            return {
                "style": "Empathetic & De-escalating",
                "empathy_statement": "I completely understand your frustration and apologize for the trouble you have experienced.",
                "guidelines": "Acknowledge the customer's inconvenience first with authentic empathy. Be concise, apologize sincerely, and prioritize step-by-step resolution without defensive explanations."
            }
        elif sentiment == "positive":
            return {
                "style": "Warm & Appreciative",
                "empathy_statement": "Thank you for reaching out! We're glad to assist you today.",
                "guidelines": "Maintain a warm, enthusiastic, and professional tone. Confirm the positive outcome directly."
            }
        else:
            return {
                "style": "Direct, Professional & Efficient",
                "empathy_statement": "Thank you for reaching out to customer support.",
                "guidelines": "Provide clear, structured, and factual instructions directly answering the customer inquiry."
            }
