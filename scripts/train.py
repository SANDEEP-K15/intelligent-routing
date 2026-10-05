"""Fit the routing model on all labelled requests and save models/router.joblib.

Usage:  python -m scripts.train [--spec reports/selected_spec.json]

The spec defaults to reports/selected_spec.json (written by scripts.evaluate) and falls back to
kestrel_router.model.DEFAULT_SPEC. Requires the assignment data in the project root.
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

from kestrel_router import __version__, config
from kestrel_router.data import load_train
from kestrel_router.model import DEFAULT_SPEC, ModelSpec, RouterModel, build_pipeline, save_model


def load_spec(path: Path) -> ModelSpec:
    if path.exists():
        return ModelSpec.from_dict(json.loads(path.read_text()))
    print(f"{path} not found; using DEFAULT_SPEC")
    return DEFAULT_SPEC


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", type=Path, default=config.REPORTS_DIR / "selected_spec.json")
    parser.add_argument("--out", type=Path, default=config.MODEL_PATH)
    args = parser.parse_args()

    spec = load_spec(args.spec)
    df = load_train()
    model = RouterModel(build_pipeline(spec)).fit(df, df["team"])

    validation = {}
    metrics_path = config.REPORTS_DIR / "metrics.json"
    if metrics_path.exists():
        sel = json.loads(metrics_path.read_text())["selected_model"]
        validation = {
            fold: {"accuracy": round(v["accuracy"], 4), "macro_f1": round(v["macro_f1"], 4)}
            for fold, v in sel.items()
        }

    model.metadata = {
        "package_version": __version__,
        "trained_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "target": "team_label (bot queue at creation), renamed teams normalised to current names",
        "spec": spec.as_dict(),
        "input_fields": ["request_text", *spec.categorical],
        "training_rows": len(df),
        "training_period": [str(df.created_at.min()), str(df.created_at.max())],
        "classes": model.classes,
        "validation": validation,
    }
    path = save_model(model, args.out)
    print(f"saved {path} ({path.stat().st_size / 1024:.0f} KiB) spec={spec.name} rows={len(df)}")


if __name__ == "__main__":
    main()
