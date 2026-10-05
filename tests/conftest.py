"""Synthetic fixtures: tests never need the confidential assignment data."""

from __future__ import annotations

import random

import pandas as pd
import pytest

from kestrel_router import config
from kestrel_router.data import add_model_inputs
from kestrel_router.model import ModelSpec, RouterModel, build_pipeline, save_model

PHRASES = {
    "Repairs": ["{p} not turning on", "{p} making loud noise", "display of {p} gone blank", "{p} tripping the mcb"],
    "Billing": ["emi conversion not done for {p}", "charged twice for my {p} order", "gst invoice wrong name for {p}"],
    "Product Advice": ["how to clean {p}", "is {p} safe for kids", "power consumption of {p}"],
    "Returns & Replacement": ["{p} arrived damaged", "want to return the {p}", "wrong model of {p} delivered"],
    "Warranty Claims": ["is {p} covered under shield plan", "claim status for {p} under warranty"],
    "Filters & Consumables": ["want amc filter kit for {p}", "need replacement filter for {p}", "spare blade set for {p}"],
    "Installs & Demo": ["please book demo and installation for {p}", "installer did not turn up for {p}"],
}
PRODUCTS = {
    "Water Purifier": "water purifier", "Air Fryer": "air fryer", "Ceiling Fan": "ceiling fan",
    "Room Heater": "room heater",
}
GREETINGS = ["", "hi, ", "good morning, ", "pls help - "]


def make_requests(n_per_team: int = 24, seed: int = 0, labelled: bool = True) -> pd.DataFrame:
    rng = random.Random(seed)
    rows = []
    i = 0
    for team, phrases in PHRASES.items():
        for _ in range(n_per_team):
            fam = rng.choice(list(PRODUCTS))
            text = rng.choice(GREETINGS) + rng.choice(phrases).format(p=PRODUCTS[fam])
            # Historical labels before the rename use the old names.
            month = 4 + (i % 9)  # Apr–Dec 2025
            label = {v: k for k, v in config.RENAME_MAP.items()}.get(team, team)
            row = {
                "request_id": f"SR{600000 + i}",
                "created_at_ist": f"2025-{month:02d}-{1 + i % 27:02d} {i % 24:02d}:{i % 60:02d}",
                "channel": rng.choice(config.CHANNELS),
                "product_family": fam,
                "warranty_status": rng.choice(config.WARRANTY_STATUSES),
                "request_text": text,
                "source": "crm",
            }
            if labelled:
                row["team_label"] = label
            rows.append(row)
            i += 1
    df = pd.DataFrame(rows).sort_values("created_at_ist").reset_index(drop=True)
    return df[list(config.TRAIN_COLUMNS if labelled else config.TEST_COLUMNS)]


@pytest.fixture(scope="session")
def synthetic_train() -> pd.DataFrame:
    df = add_model_inputs(make_requests())
    from kestrel_router.data import normalize_team

    df["team"] = df["team_label"].map(normalize_team)
    return df


@pytest.fixture(scope="session")
def trained_model(synthetic_train) -> RouterModel:
    spec = ModelSpec("test_svc_cal", ("product_family",), "svc_cal", 1.0)
    model = RouterModel(build_pipeline(spec)).fit(synthetic_train, synthetic_train["team"])
    model.metadata = {
        "spec": spec.as_dict(),
        "input_fields": ["request_text", "product_family"],
        "training_rows": len(synthetic_train),
        "trained_at_utc": "test",
    }
    return model


@pytest.fixture(scope="session")
def model_path(trained_model, tmp_path_factory):
    path = tmp_path_factory.mktemp("model") / "router.joblib"
    save_model(trained_model, path)
    return path
