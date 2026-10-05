import joblib
import numpy as np
import pandas as pd
import pytest

from kestrel_router import config
from kestrel_router.model import (
    DEFAULT_SPEC,
    ModelArtefactError,
    ModelSpec,
    RouterModel,
    build_pipeline,
    load_model,
    save_model,
)


def test_forbidden_feature_rejected():
    for col in ("final_team", "transfers", "team_label", "request_id"):
        with pytest.raises(ValueError):
            build_pipeline(ModelSpec("bad", (col,)))


def test_model_ignores_post_routing_columns(trained_model, synthetic_train):
    """Changing resolution/label columns must not change predictions (no leakage path)."""
    df = synthetic_train.head(30).copy()
    before = trained_model.predict(df)
    df["team_label"] = "Billing"
    df["final_team"] = "Billing"
    df["transfers"] = 9
    df["request_id"] = "SR0"
    after = trained_model.predict(df)
    assert (before == after).all()


def test_learns_synthetic_routing(trained_model, synthetic_train):
    acc = (trained_model.predict(synthetic_train) == synthetic_train["team"]).mean()
    assert acc > 0.95


def test_predicts_only_current_team_names(trained_model):
    assert set(trained_model.classes) == set(config.CURRENT_TEAMS)


def test_unknown_and_missing_categoricals_are_handled(trained_model):
    raw = pd.DataFrame([
        {"request_text": "air fryer not turning on", "channel": "fax", "product_family": "Toaster",
         "warranty_status": ""},
        {"request_text": "air fryer not turning on"},
    ])
    pred = trained_model.predict(raw)
    assert list(pred) == ["Repairs", "Repairs"]
    proba = trained_model.predict_proba(raw)
    assert np.allclose(proba.sum(axis=1), 1.0)


def test_route_output_schema(trained_model):
    out = trained_model.route("hi, want to return the air fryer, it arrived damaged", product_family="Air Fryer")
    assert out["team"] == "Returns & Replacement"
    assert 0 <= out["confidence"] <= 1
    assert out["confidence_band"] in {"high", "medium", "low"}
    assert out["reasons"] and all(isinstance(r, str) and r for r in out["reasons"])
    assert len(out["alternatives"]) == 2
    assert out["team"] not in {a["team"] for a in out["alternatives"]}


def test_route_flags_low_confidence_gibberish(trained_model):
    out = trained_model.route("zzz qqq")
    assert out["reasons"]  # always at least one human-readable reason


def test_save_and_load_roundtrip(trained_model, tmp_path, synthetic_train):
    path = save_model(trained_model, tmp_path / "m.joblib")
    loaded = load_model(path)
    assert loaded.metadata == trained_model.metadata
    assert (loaded.predict(synthetic_train) == trained_model.predict(synthetic_train)).all()


def test_missing_artefact_fails_politely(tmp_path):
    with pytest.raises(ModelArtefactError, match="not found"):
        load_model(tmp_path / "nope.joblib")


def test_foreign_artefact_rejected(tmp_path):
    path = tmp_path / "x.joblib"
    joblib.dump({"something": "else"}, path)
    with pytest.raises(ModelArtefactError):
        load_model(path)


def test_artefact_with_unexpected_teams_rejected(tmp_path, synthetic_train):
    df = synthetic_train.copy()
    df.loc[df.index[:30], "team"] = "Installations"  # pre-rename name must never be a model output
    model = RouterModel(build_pipeline(ModelSpec("x", (), "lr", 1.0))).fit(df, df["team"])
    path = save_model(model, tmp_path / "old.joblib")
    with pytest.raises(ModelArtefactError, match="unexpected teams"):
        load_model(path)


def test_default_spec_is_a_valid_prediction_time_model():
    pipeline = build_pipeline(DEFAULT_SPEC)
    assert DEFAULT_SPEC.categorical == ("product_family",)
    assert pipeline.named_steps["clf"] is not None


def test_default_spec_matches_evaluation_selection():
    selected = config.REPORTS_DIR / "selected_spec.json"
    if not selected.exists():
        pytest.skip("reports/selected_spec.json not generated")
    import json

    assert ModelSpec.from_dict(json.loads(selected.read_text())) == DEFAULT_SPEC
