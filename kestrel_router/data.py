"""Loading, schema validation and normalisation of the assignment files.

The original files are opened read-only and never written. Validation problems raise
DataValidationError with every issue listed; no row is dropped silently.
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from . import config
from .text import clean_text

_REQUEST_ID_RE = re.compile(r"^SR\d+$")
_TIMESTAMP_FORMAT = "%Y-%m-%d %H:%M"


class DataValidationError(ValueError):
    """Raised when an input file does not match the documented schema."""

    def __init__(self, source: str, problems: list[str]):
        self.source = source
        self.problems = problems
        super().__init__(f"{source}: " + "; ".join(problems))


def read_csv(path: Path | str) -> pd.DataFrame:
    """Read a CSV as strings without coercing empty cells to NaN."""
    return pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8")


def normalize_team(label: str) -> str:
    """Map a historical team label to its current name (15 Jan 2026 rename)."""
    if label in config.CURRENT_TEAMS:
        return label
    if label in config.RENAME_MAP:
        return config.RENAME_MAP[label]
    raise DataValidationError("team label", [f"unknown team {label!r}"])


def _check_columns(df: pd.DataFrame, expected: tuple[str, ...], source: str) -> None:
    if tuple(df.columns) != expected:
        raise DataValidationError(
            source, [f"expected columns {list(expected)}, found {list(df.columns)}"]
        )


def _check_categories(
    df: pd.DataFrame, column: str, allowed: tuple[str, ...], problems: list[str]
) -> None:
    bad = df.loc[~df[column].isin(allowed), column]
    if len(bad):
        problems.append(
            f"{len(bad)} rows with unexpected {column} values {sorted(bad.unique())[:5]}"
        )


def validate_requests(df: pd.DataFrame, source: str, labelled: bool) -> pd.DataFrame:
    """Validate a train/test request frame and return it with parsed timestamps."""
    _check_columns(df, config.TRAIN_COLUMNS if labelled else config.TEST_COLUMNS, source)
    problems: list[str] = []

    if df.empty:
        problems.append("file has no rows")
    bad_ids = df.loc[~df["request_id"].str.match(_REQUEST_ID_RE), "request_id"]
    if len(bad_ids):
        problems.append(f"{len(bad_ids)} malformed request_id values, e.g. {bad_ids.iloc[0]!r}")
    dup = df["request_id"].duplicated()
    if dup.any():
        problems.append(f"{int(dup.sum())} duplicate request_id values")

    created = pd.to_datetime(df["created_at_ist"], format=_TIMESTAMP_FORMAT, errors="coerce")
    if created.isna().any():
        problems.append(f"{int(created.isna().sum())} unparseable created_at_ist values")

    empty_text = df["request_text"].str.strip() == ""
    if empty_text.any():
        problems.append(f"{int(empty_text.sum())} empty request_text values")

    _check_categories(df, "channel", config.CHANNELS, problems)
    _check_categories(df, "warranty_status", config.WARRANTY_STATUSES, problems)
    _check_categories(df, "product_family", config.PRODUCT_FAMILIES, problems)
    _check_categories(df, "source", config.SOURCES, problems)

    if labelled:
        _check_categories(df, "team_label", config.HISTORICAL_TEAMS, problems)
        if not created.isna().any():
            old = df["team_label"].isin(list(config.RENAME_MAP))
            after = created >= pd.Timestamp(config.RENAME_DATE)
            new = df["team_label"].isin(list(config.RENAME_MAP.values()))
            if (old & after).any():
                problems.append(f"{int((old & after).sum())} pre-rename team names after {config.RENAME_DATE}")
            if (new & ~after).any():
                problems.append(f"{int((new & ~after).sum())} post-rename team names before {config.RENAME_DATE}")

    if problems:
        raise DataValidationError(source, problems)
    out = df.copy()
    out["created_at"] = created
    return out


def validate_resolution_log(df: pd.DataFrame, source: str = config.RESOLUTION_FILE) -> pd.DataFrame:
    _check_columns(df, config.RESOLUTION_COLUMNS, source)
    problems: list[str] = []
    if df["request_id"].duplicated().any():
        problems.append("duplicate request_id values")
    for col in ("first_team", "final_team"):
        _check_categories(df, col, config.HISTORICAL_TEAMS, problems)
    if not df["transfers"].str.fullmatch(r"\d+").all():
        problems.append("non-integer transfers values")
    resolved = pd.to_datetime(df["resolved_at"], format=_TIMESTAMP_FORMAT, errors="coerce")
    if resolved.isna().any():
        problems.append(f"{int(resolved.isna().sum())} unparseable resolved_at values")
    if problems:
        raise DataValidationError(source, problems)
    out = df.copy()
    out["transfers"] = out["transfers"].astype(int)
    out["resolved_at_raw"] = resolved
    return out


def validate_teams_file(df: pd.DataFrame, source: str = config.TEAMS_FILE) -> pd.DataFrame:
    _check_columns(df, ("team", "renamed_to", "handles"), source)
    current = [
        row.renamed_to.split(" (from")[0] if row.renamed_to else row.team
        for row in df.itertuples()
    ]
    if sorted(current) != sorted(config.CURRENT_TEAMS):
        raise DataValidationError(
            source, [f"team list {sorted(current)} != expected {sorted(config.CURRENT_TEAMS)}"]
        )
    return df


def add_model_inputs(df: pd.DataFrame) -> pd.DataFrame:
    """Add the cleaned text column used by the model. Categorical columns pass through."""
    out = df.copy()
    out["text_clean"] = out["request_text"].map(clean_text)
    for col in ("channel", "warranty_status", "product_family"):
        if col not in out:
            out[col] = config.UNKNOWN
        out[col] = out[col].replace("", config.UNKNOWN).fillna(config.UNKNOWN)
    return out


def load_train(data_dir: Path = config.DATA_DIR) -> pd.DataFrame:
    """Labelled requests with normalised `team` plus `final_team` (evaluation only)."""
    train = validate_requests(read_csv(data_dir / config.TRAIN_FILE), config.TRAIN_FILE, labelled=True)
    log = validate_resolution_log(read_csv(data_dir / config.RESOLUTION_FILE))
    validate_teams_file(read_csv(data_dir / config.TEAMS_FILE))

    missing = set(train["request_id"]) - set(log["request_id"])
    extra = set(log["request_id"]) - set(train["request_id"])
    if missing or extra:
        raise DataValidationError(
            config.RESOLUTION_FILE,
            [f"{len(missing)} train ids missing from log, {len(extra)} log ids not in train"],
        )

    train = train.merge(log, on="request_id", how="left", validate="one_to_one")
    train["team"] = train["team_label"].map(normalize_team)
    train["final_team_norm"] = train["final_team"].map(normalize_team)
    train["first_team_norm"] = train["first_team"].map(normalize_team)
    return add_model_inputs(train)


def load_test(data_dir: Path = config.DATA_DIR) -> pd.DataFrame:
    test = validate_requests(read_csv(data_dir / config.TEST_FILE), config.TEST_FILE, labelled=False)
    return add_model_inputs(test)


def time_split(df: pd.DataFrame, fold: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    spec = config.FOLDS[fold]
    train = df[df["created_at"] < pd.Timestamp(spec["train_end"])]
    val = df[
        (df["created_at"] >= pd.Timestamp(spec["val_start"]))
        & (df["created_at"] < pd.Timestamp(spec["val_end"]))
    ]
    return train, val
