# Operations handoff

## What runs in production

- **Service:** a single Python process, `uvicorn service.app:app`, on any machine with Python 3.11+.
- **Model:** the file `models/router.joblib` (~205 KB, loaded once into memory).
- **Not needed:** no database, GPU, network access or API key.
- **Scaling:** the service is stateless, so to scale you run more copies behind a load balancer.

```bash
pip install -r requirements.txt
uvicorn service.app:app --host 0.0.0.0 --port 8000
python -m scripts.smoke_test --url http://localhost:8000
```

`GET /api/health` reports which model is loaded and its validation scores. If the artefact is
missing, the service still starts: health returns `model_missing` and routing returns 503.

## Recommended rollout: shadow first

1. **Shadow (2–4 weeks).**
   - Call `/api/route` for every new request alongside the vendor bot. Log both teams; agents keep
     using the bot's queue.
   - Agreement with the bot on live traffic should stay at or above 90% (expected ~96%).
   - Also record, once requests close, how often each choice matched the closing team.
2. **Switch.** Route by the model and keep the bot licence until the end of its current term.
3. **Fallback.** For `confidence_band = "low"` (confidence below 0.5), send the request to the
   service-desk triage queue, or ask the customer to clarify, rather than to the suggested team.

## Monitoring (monthly)

Calculate these from closed requests:

| Check | Source | Alert if |
|---|---|---|
| Model vs bot label (during shadow) | shadow log | < 90% |
| Model vs closing team | resolution log | drops > 3 points from the 77% baseline |
| Share of low-confidence predictions | service log | rises > 5 points |
| Team volume mix | predictions | any team moves > 5 points |

## Retraining

Retrain when monitoring alerts fire, when team responsibilities change, or quarterly.

```bash
# assignment-format CSVs in the project root
python -m scripts.evaluate     # time-based comparison and reports
python -m scripts.train        # fits reports/selected_spec.json on all labelled data
python -m scripts.release_check
```

- **New team names:** a team rename needs a `config.RENAME_MAP` entry and the current names in
  `config.CURRENT_TEAMS`. Data validation fails loudly until both are updated.
- **Versions:** keep `requirements.txt` pinned. The artefact must be loaded with the same
  scikit-learn version it was built with, and the service warns if they differ.

## Data handling

- **Never publish:** the request text, labels, the resolution log, `predictions.csv` and
  `router.joblib` are not to be published (ops policy §10).
- **Before every push:** `python -m scripts.audit_public_repo` must pass. CI also runs it on every
  push.
