"""Evaluation metrics and cost arithmetic (ops policy §4 figures)."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support

from . import config


def score(y_true: Sequence[str], y_pred: Sequence[str]) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(
            f1_score(y_true, y_pred, average="macro", labels=list(config.CURRENT_TEAMS), zero_division=0)
        ),
        "n": len(y_true),
    }


def per_team(y_true: Sequence[str], y_pred: Sequence[str]) -> pd.DataFrame:
    labels = list(config.CURRENT_TEAMS)
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    return pd.DataFrame({"precision": p, "recall": r, "f1": f, "support": s}, index=labels)


def confusion(y_true: Sequence[str], y_pred: Sequence[str]) -> pd.DataFrame:
    labels = list(config.CURRENT_TEAMS)
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    return pd.DataFrame(cm, index=[f"true: {t}" for t in labels], columns=labels)


def misroute_cost(n_misrouted: int, avg_transfers_per_misroute: float) -> float:
    """Rs cost of misroutes: transfers × Rs 305 plus one extra contact × Rs 260 each."""
    per = avg_transfers_per_misroute * config.COST_PER_TRANSFER + config.COST_EXTRA_CONTACT_PER_MISROUTE
    return n_misrouted * per


def monthly_model_cost(requests_per_month: int) -> dict[str, Any]:
    return {
        "requests_per_month": requests_per_month,
        "cost_per_prediction_rs": config.MODEL_API_COST_PER_PREDICTION,
        "monthly_model_api_cost_rs": requests_per_month * config.MODEL_API_COST_PER_PREDICTION,
        "bot_licence_per_month_rs": round(config.BOT_LICENCE_PER_YEAR / 12, 2),
    }
