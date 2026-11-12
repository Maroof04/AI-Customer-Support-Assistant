"""
Text Preprocessing Module for Customer Support Assistant.
Provides clean, consistent string transformations and entity/metadata extraction.
"""

import re
from typing import Dict, Any, Optional

# Common English contractions dictionary for normalization
CONTRACTIONS_MAP = {
    r"\bcan't\b": "cannot",
    r"\bwon't\b": "will not",
    r"\bdoesn't\b": "does not",
    r"\bdon't\b": "do not",
    r"\bdidn't\b": "did not",
    r"\bisn't\b": "is not",
    r"\baren't\b": "are not",
    r"\bwasn't\b": "was not",
    r"\bweren't\b": "were not",
    r"\bhaven't\b": "have not",
    r"\bhasn't\b": "has not",
    r"\bhadn't\b": "had not",
    r"\bwouldn't\b": "would not",
    r"\bshouldn't\b": "should not",
    r"\bcouldn't\b": "could not",
    r"\bi'm\b": "i am",
    r"\byou're\b": "you are",
    r"\bwe're\b": "we are",
    r"\bthey're\b": "they are",
    r"\bi've\b": "i have",
    r"\byou've\b": "you have",
    r"\bi'll\b": "i will",
    r"\byou'll\b": "you will",
    r"\bit's\b": "it is",
    r"\bthat's\b": "that is",
    r"\bwhat's\b": "what is",
    r"\bwhere's\b": "where is",
}


def clean_text(text: str) -> str:
    """
    Cleans raw customer query text for NLP modeling.
    - Lowercases text
    - Expands standard contractions
    - Normalizes punctuation and excessive whitespace
    - Strips unwanted non-alphanumeric noise while preserving core terms
    """
    if not isinstance(text, str):
        return ""

    text = text.lower().strip()

    # Expand contractions
    for pattern, replacement in CONTRACTIONS_MAP.items():
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)

    # Replace URL links
    text = re.sub(r"https?://\S+|www\.\S+", " [url] ", text)

    # Standardize numbers and codes with spaces
    text = re.sub(r"#(\d+)", r" order_\1 ", text)

    # Keep only letters, digits, and basic punctuation
    text = re.sub(r"[^a-zA-Z0-9\s\.\?!_\$]", " ", text)

    # Collapse multiple whitespace characters into single space
    text = re.sub(r"\s+", " ", text).strip()

    return text


def extract_metadata(text: str) -> Dict[str, Any]:
    """
    Extracts structured entities commonly found in customer support queries:
    - Order numbers (e.g., #84910, order 12345)
    - Tracking or RMA numbers
    - Currency amounts (e.g., $89.99, $15)
    - Error codes (e.g., 500, 502, ERR_...)
    """
    if not isinstance(text, str):
        return {"order_ids": [], "amounts": [], "error_codes": []}

    order_ids = re.findall(r"#\d+|order\s*#?\s*\d+", text, flags=re.IGNORECASE)
    amounts = re.findall(r"\$\s*\d+(?:\.\d{2})?", text)
    error_codes = re.findall(r"\b(?:error\s*\d{3}|err_[a-z0-9_]+|\d{3}\s*error)\b", text, flags=re.IGNORECASE)

    return {
        "order_ids": [oid.strip() for oid in order_ids],
        "amounts": [amt.strip() for amt in amounts],
        "error_codes": [ec.strip() for ec in error_codes]
    }
