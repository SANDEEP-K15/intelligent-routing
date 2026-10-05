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
