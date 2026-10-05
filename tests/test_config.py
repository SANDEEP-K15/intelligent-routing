"""Invariants of the shared configuration."""

from kestrel_router import config


def test_seven_current_teams():
    assert len(config.CURRENT_TEAMS) == 7
    assert len(set(config.CURRENT_TEAMS)) == 7


def test_rename_map_targets_current_names():
    assert set(config.RENAME_MAP.values()) <= set(config.CURRENT_TEAMS)
    assert not set(config.RENAME_MAP) & set(config.CURRENT_TEAMS)
    assert set(config.HISTORICAL_TEAMS) == set(config.CURRENT_TEAMS) | set(config.RENAME_MAP)


def test_post_routing_columns_are_forbidden_features():
    for col in ("team_label", "first_team", "final_team", "transfers", "resolved_at", "request_id"):
        assert col in config.FORBIDDEN_FEATURES


def test_folds_are_chronological_and_non_overlapping():
    for fold in config.FOLDS.values():
        assert fold["train_end"] <= fold["val_start"] < fold["val_end"]
    assert config.FOLDS["backtest"]["val_end"] <= config.FOLDS["primary"]["val_start"]


def test_schemas():
    assert config.TEST_COLUMNS == config.TRAIN_COLUMNS[:-1]
    assert config.TRAIN_COLUMNS[-1] == "team_label"
    assert config.SUBMISSION_COLUMNS == ("request_id", "team")


def test_policy_cost_figures():
    assert config.COST_PER_TRANSFER == 305
    assert config.COST_EXTRA_CONTACT_PER_MISROUTE == 260
    assert config.BOT_LICENCE_PER_YEAR == 320_000
    assert config.MODEL_API_COST_PER_PREDICTION == 0
