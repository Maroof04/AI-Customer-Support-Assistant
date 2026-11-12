"""
Unit tests for text preprocessing and entity extraction.
"""

import pytest
from src.preprocessing import clean_text, extract_metadata


def test_clean_text_normalizes_case():
    raw = "PLEASE REFUND MY MONEY NOW"
    cleaned = clean_text(raw)
    assert cleaned == "please refund my money now"


def test_clean_text_expands_contractions():
    raw = "I can't access my account and shouldn't have been charged."
    cleaned = clean_text(raw)
    assert "cannot" in cleaned
    assert "should not" in cleaned


def test_clean_text_handles_special_characters():
    raw = "Order #99281 was broken!!! @#$$"
    cleaned = clean_text(raw)
    assert "order_99281" in cleaned
    assert "@" not in cleaned


def test_extract_metadata_order_and_amounts():
    text = "Order #84910 was charged $89.99 twice. Need order #84910 refunded."
    meta = extract_metadata(text)
    assert len(meta["order_ids"]) >= 1
    assert "$89.99" in meta["amounts"]


def test_extract_metadata_error_codes():
    text = "Encountered Error 500 and ERR_CARD_AUTH_FAILED at checkout."
    meta = extract_metadata(text)
    assert len(meta["error_codes"]) >= 1


def test_clean_text_empty_input():
    assert clean_text("") == ""
    assert clean_text(None) == ""
