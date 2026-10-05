"""Constants shared by the pipeline, model, scripts and service."""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT  # the original assignment files sit in the project root (read-only)
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
PRIVATE_REPORTS_DIR = REPORTS_DIR / "private"
MODEL_PATH = MODELS_DIR / "router.joblib"

TRAIN_FILE = "train.csv"
TEST_FILE = "test_unlabelled.csv"
RESOLUTION_FILE = "resolution_log.csv"
TEAMS_FILE = "teams.csv"
SAMPLE_SUBMISSION_FILE = "sample_submission.csv"

# Ops policy §5 / teams.csv: renamed on 15 Jan 2026, responsibilities unchanged.
RENAME_DATE = "2026-01-15"
RENAME_MAP = {
    "Installations": "Installs & Demo",
    "Consumables": "Filters & Consumables",
}

# The seven current team names (output vocabulary for predictions.csv and the API).
CURRENT_TEAMS = (
    "Repairs",
    "Billing",
    "Product Advice",
    "Returns & Replacement",
    "Warranty Claims",
    "Filters & Consumables",
    "Installs & Demo",
)
HISTORICAL_TEAMS = tuple(sorted(set(CURRENT_TEAMS) | set(RENAME_MAP)))

CHANNELS = ("ivr", "chat", "whatsapp", "email")
WARRANTY_STATUSES = ("in_warranty", "shield", "out_of_warranty")
PRODUCT_FAMILIES = (
    "Water Purifier",
    "Air Fryer",
    "Mixer Grinder",
    "Induction Cooktop",
    "Room Heater",
    "Ceiling Fan",
    "Robot Vacuum",
)
SOURCES = ("crm", "legacy_zoho")
UNKNOWN = "unknown"

TRAIN_COLUMNS = (
    "request_id",
    "created_at_ist",
    "channel",
    "product_family",
    "warranty_status",
    "request_text",
    "source",
    "team_label",
)
TEST_COLUMNS = TRAIN_COLUMNS[:-1]
RESOLUTION_COLUMNS = ("request_id", "first_team", "final_team", "transfers", "resolved_at")
SUBMISSION_COLUMNS = ("request_id", "team")

# Columns that only exist after routing/resolution. Never allowed as model features.
FORBIDDEN_FEATURES = (
    "team_label",
    "first_team",
    "final_team",
    "transfers",
    "resolved_at",
    "request_id",
    "created_at_ist",
    "source",
)

# Validation folds (D3 in docs/DECISIONS.md). Intervals are [start, end).
FOLDS = {
    "primary": {"train_end": "2026-04-01", "val_start": "2026-04-01", "val_end": "2026-07-01"},
    "backtest": {"train_end": "2026-01-01", "val_start": "2026-01-01", "val_end": "2026-04-01"},
}

# Ops policy §4 cost figures (Rs).
COST_PER_TRANSFER = 305
COST_EXTRA_CONTACT_PER_MISROUTE = 260
BOT_LICENCE_PER_YEAR = 320_000
MODEL_API_COST_PER_PREDICTION = 0.0
