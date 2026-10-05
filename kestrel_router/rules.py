"""Deterministic keyword baseline.

Built from the team descriptions in teams.csv plus the three routing habits of the existing bot
identified in discovery. It is a comparison point; it is not used in production.
"""

from __future__ import annotations

import re

import pandas as pd

_VAGUE = re.compile(
    r"someone contact me|not happy with|complaint about|need help with|please call back regarding"
    r"|service request for|\bquery\b|\bproblem\b|issue with|^(\w+\W+){0,3}help\b"
)
_PAID = re.compile(r"\bpaid\b|payment done")
_FAULT = re.compile(
    r"tripping|leak|gone blank|noise|burnt|not turning on|error code|stopped working|not working"
)

# Ordered: first match wins. Keyword lists paraphrase the teams.csv "handles" column.
_TEAM_RULES: list[tuple[str, re.Pattern[str]]] = [
    ("Billing", re.compile(r"invoice|gst|charged twice|double charge|refund|emi|coupon|bill\b|payment deducted")),
    ("Returns & Replacement", re.compile(r"return|exchange|damaged|scratched|wrong model|missing parts|box was open|received used|need replacement(?! filter)")),
    ("Warranty Claims", re.compile(r"warranty|shield")),
    ("Installs & Demo", re.compile(r"install|demo|wall mounting|installer")),
    ("Filters & Consumables", re.compile(r"filter|candle|membrane|jar|brush|blade|amc|consumable|spare")),
    ("Repairs", _FAULT),
    ("Product Advice", re.compile(r"how to|safe for|power consumption|settings|inverter|difference between|which .* right|recipe|clean")),
]


def predict_one(text_clean: str, product_family: str) -> str:
    is_purifier = product_family == "Water Purifier"
    if _PAID.search(text_clean):
        return "Billing"
    if is_purifier and (_FAULT.search(text_clean) or _VAGUE.search(text_clean)):
        return "Filters & Consumables"
    for team, pattern in _TEAM_RULES:
        if pattern.search(text_clean):
            return team
    return "Repairs"


def predict(df: pd.DataFrame) -> list[str]:
    return [predict_one(t, p) for t, p in zip(df["text_clean"], df["product_family"], strict=False)]
