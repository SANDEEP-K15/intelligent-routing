"""Submission-file validation: format must match sample_submission.csv exactly."""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd

from . import config


def validate_submission(
    submission: pd.DataFrame,
    test_ids: Sequence[str],
    sample: pd.DataFrame | None = None,
) -> list[str]:
    """Return a list of problems (empty list = valid)."""
    problems: list[str] = []
    expected_cols = list(config.SUBMISSION_COLUMNS)
    if list(submission.columns) != expected_cols:
        problems.append(f"columns {list(submission.columns)} != {expected_cols}")
        return problems
    if sample is not None and list(sample.columns) != expected_cols:
        problems.append(f"sample_submission columns {list(sample.columns)} != {expected_cols}")

    test_ids = list(test_ids)
    if len(submission) != len(test_ids):
        problems.append(f"row count {len(submission)} != test rows {len(test_ids)}")
    ids = submission["request_id"]
    if ids.isna().any() or (ids.astype(str).str.strip() == "").any():
        problems.append("missing request_id values")
    if ids.duplicated().any():
        problems.append(f"{int(ids.duplicated().sum())} duplicate request_id values")
    missing = set(test_ids) - set(ids)
    extra = set(ids) - set(test_ids)
    if missing:
        problems.append(f"{len(missing)} test request_ids have no prediction")
    if extra:
        problems.append(f"{len(extra)} request_ids are not in the test file")
    if not missing and not extra and list(ids) != test_ids:
        problems.append("request_id order differs from test_unlabelled.csv")
    if sample is not None and list(sample["request_id"]) != list(ids):
        problems.append("request_id order differs from sample_submission.csv")

    teams = submission["team"]
    if teams.isna().any() or (teams.astype(str).str.strip() == "").any():
        problems.append("missing team predictions")
    bad = sorted(set(teams.dropna()) - set(config.CURRENT_TEAMS))
    if bad:
        problems.append(f"teams outside the seven current names: {bad}")
    return problems
