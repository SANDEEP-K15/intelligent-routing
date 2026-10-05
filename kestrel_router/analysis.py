"""Request-pattern tagging used for error analysis and reporting (not used as model features)."""

from __future__ import annotations

import re

import pandas as pd

_GREETING = re.compile(r"^(good morning|hello team|namaste|hi|hello|sir|urgent|pls help)\W*\s*")
_PRODUCT = re.compile(
    r"\b(water purifier|air fryer|mixer grinder|induction cooktop|room heater|ceiling fan|robot vacuum"
    r"|purifier|fryer|grinder|cooktop|heater|fan|vacuum|mixer|product|machine)\b"
)
_SIGN_OFF = re.compile(r"\b(kindly resolve|very disappointed|pls call back|thanks|asap)\b")
_PAID = re.compile(r"\bpaid\b|payment done")
_VAGUE = re.compile(
    r"someone contact me|not happy with|complaint about|need help with|please call back regarding"
    r"|service request for|\bquery\b|\bproblem\b|issue with|^(\w+\W+){0,3}help (<p>)"
)
_FAULT = re.compile(
    r"tripping|leak|gone blank|noise|burnt|not turning on|error code|stopped working|not working"
)


def template(text_clean: str) -> str:
    """Strip greetings, sign-offs and product names so wording patterns can be counted
    without exposing individual customer messages."""
    t = text_clean
    for _ in range(3):
        t = _GREETING.sub("", t).strip()
    t = _SIGN_OFF.sub(" ", t)
    t = re.sub(r"\b(reg no|order)\b", " ", t)
    t = _PRODUCT.sub("<p>", t)
    t = re.sub(r"(<p>\s*)+<p>", "<p>", t)
    return re.sub(r"\s+", " ", t).strip(" ,.-:")


def pattern_group(text_clean: str, product_family: str) -> str:
    """Mutually exclusive discovery groups: paid > purifier fault > vague > clear."""
    t = template(text_clean)
    if _PAID.search(text_clean):
        return "mentions payment"
    if product_family == "Water Purifier" and _FAULT.search(text_clean):
        return "purifier fault"
    if _VAGUE.search(t):
        return "vague"
    return "clear"


def is_multi_intent(text_clean: str) -> bool:
    return "," in template(text_clean)


def tag(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["template"] = out["text_clean"].map(template)
    out["pattern"] = [pattern_group(t, p) for t, p in zip(out["text_clean"], out["product_family"], strict=False)]
    out["multi_intent"] = out["text_clean"].map(is_multi_intent)
    return out
