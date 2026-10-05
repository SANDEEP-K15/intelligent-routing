import pandas as pd
import pytest

from kestrel_router import config
from kestrel_router.submission import validate_submission

IDS = [f"SR{510822 + i}" for i in range(5)]


def good() -> pd.DataFrame:
    return pd.DataFrame({"request_id": IDS, "team": list(config.CURRENT_TEAMS[:5])})


def sample() -> pd.DataFrame:
    return pd.DataFrame({"request_id": IDS, "team": ["Repairs"] * 5})


def test_valid_submission():
    assert validate_submission(good(), IDS, sample()) == []


@pytest.mark.parametrize(
    "mutate, message",
    [
        (lambda d: d.assign(extra=1), "columns"),
        (lambda d: d.rename(columns={"team": "team_label"}), "columns"),
        (lambda d: d.iloc[:4], "row count"),
        (lambda d: d.assign(request_id=[IDS[0]] + IDS[:4]), "duplicate"),
        (lambda d: d.assign(team=["Installations"] + list(d.team[1:])), "outside the seven"),
        (lambda d: d.assign(team=[""] + list(d.team[1:])), "missing team"),
        (lambda d: d.assign(request_id=["SR1"] + IDS[1:]), "not in the test file"),
        (lambda d: d.iloc[::-1].reset_index(drop=True), "order"),
    ],
)
def test_invalid_submissions(mutate, message):
    problems = validate_submission(mutate(good()), IDS, sample())
    assert any(message in p for p in problems), problems
