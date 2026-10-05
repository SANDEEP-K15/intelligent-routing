import pandas as pd
import pytest

from kestrel_router import config
from kestrel_router.data import (
    DataValidationError,
    add_model_inputs,
    load_test,
    load_train,
    normalize_team,
    time_split,
    validate_requests,
    validate_teams_file,
)
from tests.conftest import make_requests


def test_rename_mapping():
    assert normalize_team("Installations") == "Installs & Demo"
    assert normalize_team("Consumables") == "Filters & Consumables"
    for team in config.CURRENT_TEAMS:
        assert normalize_team(team) == team


def test_unknown_team_fails():
    with pytest.raises(DataValidationError):
        normalize_team("Sales")


def test_valid_frame_passes_and_keeps_every_row():
    df = make_requests()
    out = validate_requests(df, "synthetic", labelled=True)
    assert len(out) == len(df)
    assert out["request_id"].tolist() == df["request_id"].tolist()


@pytest.mark.parametrize(
    "mutate, message",
    [
        (lambda d: d.assign(channel=d.channel.where(d.index != 0, "fax")), "channel"),
        (lambda d: d.assign(request_id=d.request_id.where(d.index != 1, d.request_id[0])), "duplicate"),
        (lambda d: d.assign(created_at_ist=d.created_at_ist.where(d.index != 2, "31/12/2025")), "created_at_ist"),
        (lambda d: d.assign(request_text=d.request_text.where(d.index != 3, "   ")), "empty request_text"),
        (lambda d: d.assign(team_label=d.team_label.where(d.index != 4, "Sales")), "team_label"),
        (lambda d: d.assign(request_id=d.request_id.where(d.index != 5, "XX1")), "malformed"),
        (lambda d: d.assign(product_family=d.product_family.where(d.index != 6, "Toaster")), "product_family"),
    ],
)
def test_unexpected_input_fails_clearly(mutate, message):
    with pytest.raises(DataValidationError) as exc:
        validate_requests(mutate(make_requests()), "synthetic", labelled=True)
    assert message in str(exc.value)


def test_wrong_columns_fail():
    df = make_requests().rename(columns={"channel": "medium"})
    with pytest.raises(DataValidationError, match="expected columns"):
        validate_requests(df, "synthetic", labelled=True)


def test_old_team_name_after_rename_date_fails():
    df = make_requests().head(3).copy()
    df["created_at_ist"] = "2026-02-01 10:00"
    df["team_label"] = "Installations"
    with pytest.raises(DataValidationError, match="pre-rename"):
        validate_requests(df, "synthetic", labelled=True)


def test_add_model_inputs_fills_missing_categoricals():
    df = pd.DataFrame([{"request_text": "Fan not turning on", "channel": "", "product_family": None}])
    out = add_model_inputs(df)
    assert out.loc[0, "text_clean"] == "fan not turning on"
    assert out.loc[0, "channel"] == config.UNKNOWN
    assert out.loc[0, "product_family"] == config.UNKNOWN
    assert out.loc[0, "warranty_status"] == config.UNKNOWN


def test_time_split_boundaries():
    df = pd.DataFrame({"created_at": pd.to_datetime(
        ["2025-12-31 23:59", "2026-01-01 00:00", "2026-03-31 23:59", "2026-04-01 00:00", "2026-06-30 23:59"])})
    tr, va = time_split(df, "primary")
    assert len(tr) == 3 and len(va) == 2
    tr, va = time_split(df, "backtest")
    assert len(tr) == 1 and len(va) == 2


def test_teams_file_validation():
    teams = pd.DataFrame({
        "team": ["Installations", "Repairs", "Consumables", "Billing", "Returns & Replacement",
                 "Warranty Claims", "Product Advice"],
        "renamed_to": ["Installs & Demo (from 15 Jan 2026)", "", "Filters & Consumables (from 15 Jan 2026)",
                       "", "", "", ""],
        "handles": ["x"] * 7,
    })
    validate_teams_file(teams)
    with pytest.raises(DataValidationError):
        validate_teams_file(teams.iloc[:6])


def test_loaders_on_synthetic_files(tmp_path):
    train = make_requests()
    train.to_csv(tmp_path / config.TRAIN_FILE, index=False)
    make_requests(labelled=False).to_csv(tmp_path / config.TEST_FILE, index=False)
    pd.DataFrame({
        "request_id": train.request_id, "first_team": train.team_label, "final_team": train.team_label,
        "transfers": "0", "resolved_at": "2025-12-31 10:00",
    }).to_csv(tmp_path / config.RESOLUTION_FILE, index=False)
    pd.DataFrame({
        "team": ["Installations", "Repairs", "Consumables", "Billing", "Returns & Replacement",
                 "Warranty Claims", "Product Advice"],
        "renamed_to": ["Installs & Demo (from 15 Jan 2026)", "", "Filters & Consumables (from 15 Jan 2026)",
                       "", "", "", ""],
        "handles": ["x"] * 7,
    }).to_csv(tmp_path / config.TEAMS_FILE, index=False)

    df = load_train(tmp_path)
    assert len(df) == len(train)
    assert set(df["team"]) <= set(config.CURRENT_TEAMS)
    assert "Installations" not in set(df["team"])
    assert len(load_test(tmp_path)) == len(train)


def test_resolution_log_mismatch_fails(tmp_path):
    test_loaders_on_synthetic_files(tmp_path)
    log = pd.read_csv(tmp_path / config.RESOLUTION_FILE, dtype=str).iloc[1:]
    log.to_csv(tmp_path / config.RESOLUTION_FILE, index=False)
    with pytest.raises(DataValidationError, match="missing from log"):
        load_train(tmp_path)
