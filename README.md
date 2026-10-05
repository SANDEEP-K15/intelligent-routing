# Kestrel Home: service-request routing

This project is a local, CPU-only classifier that suggests the team queue for a new Kestrel Home
service request. It is intended to replace a licensed vendor routing bot. It uses no paid model
API.

The target and design decisions are recorded in [docs/DECISIONS.md](docs/DECISIONS.md).

## Scope

- **Input:** a customer's opening message (chat, WhatsApp, email or IVR transcript), with optional
  channel, product family and warranty status.
- **Output:** one of the seven current team queues, with a confidence and a plain-language reason.
- **Constraints:** it must start on a clean machine with Python and `pip install -r
  requirements.txt`; routing has no per-request cost.

## Setup

Python 3.11 or newer.

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS / Linux
pip install -r requirements-dev.txt
pytest
ruff check .
```

## Reproducing predictions locally

Place the assignment files in the project root; `.gitignore` keeps them out of git. Then:

```bash
python -m scripts.evaluate      # time-based model comparison; writes aggregate reports/ (about 5 min)
python -m scripts.train         # fits the selected model on all labelled requests → models/router.joblib
python -m scripts.predict       # writes predictions.csv, only if it passes validation
python -m scripts.validate_submission predictions.csv
```

- **Format:** `predictions.csv` has the format of `sample_submission.csv`: header
  `request_id,team`, one row per test request in test-file order, and only the seven current team
  names. See [examples/predictions.example.csv](examples/predictions.example.csv), which uses
  fictitious IDs.
- **Not committed:** the real file is request-level output, so it is git-ignored and shared
  privately.

## Data handling

The Kestrel assignment data is confidential. Ops policy §10 forbids publishing it or uploading it
to public repositories.

- **Excluded by `.gitignore`:** every assignment file, all CSV and PDF files, model artefacts and
  row-level outputs.
- **Checked before every push:**

  ```bash
  python -m scripts.audit_public_repo
  ```

  The audit blocks restricted files, real request or order numbers, likely secrets and (when the
  data is present locally) verbatim customer text.
