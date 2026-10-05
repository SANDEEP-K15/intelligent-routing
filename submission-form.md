# Submission form: Task 2 V2, Kestrel Home service-request routing

> Three fields are marked **UPDATE BEFORE SUBMISSION** because the links or values do not exist yet.
> Every other field is final.

## Links

| Field | Value |
|---|---|
| GitHub repository | https://github.com/SANDEEP-K15/intelligent-routing (public). It contains code, tests, documentation and aggregate metrics only; no Kestrel data, per ops policy §10. |
| Google Drive folder | **UPDATE BEFORE SUBMISSION:** the folder has not been created yet. Upload `predictions.csv` and `models/router.joblib` there with restricted sharing. |
| Screen recording | **UPDATE BEFORE SUBMISSION:** the recording has not been made yet. Follow `docs/recording-checklist.md`. |

## Effort

| Field | Value |
|---|---|
| Hours spent | **UPDATE BEFORE SUBMISSION:** hours not yet recorded by the candidate |

## Result

| Field | Value |
|---|---|
| Prediction target | `team_label` (the bot's queue at creation), with the two 15 Jan 2026 renames normalised to the current seven team names |
| Approach | Classical supervised ML: word and character TF-IDF over the cleaned request text, plus product family, with a calibrated linear SVM. Local CPU, no LLM, no paid API. |
| Validation design | Time-based. Train Apr 2025 – Mar 2026 → validate Apr–Jun 2026; backtest train Apr–Dec 2025 → validate Jan–Mar 2026. The test set was never used for tuning. |
| Validation accuracy (Apr–Jun 2026) | 96.8% (95% CI 96.1–97.6%), macro-F1 0.967 |
| Backtest accuracy (Jan–Mar 2026) | 97.0% (95% CI 96.3–97.7%), macro-F1 0.968 |
| Baselines | Majority class 28.8%; keyword rules 91.4%; logistic regression 96.3% (mean of both folds) |
| Expected hidden-test accuracy | 95.5% – 97.5%, most likely ≈ 96.5% (an estimate, not a guarantee) |
| Meets the client's 90% bar on validation? | Yes, in both folds and in every validation month (lowest month 96.2%) |
| `predictions.csv` | 2,178 rows; columns `request_id,team`; IDs match `test_unlabelled.csv` in order; only the seven current team names; checked by `scripts/validate_submission.py` |

## Cost

| Field | Value |
|---|---|
| Direct model/API cost per prediction | Rs 0 (local CPU inference) |
| Monthly model/API cost at ~700 requests/month | 700 × Rs 0 = **Rs 0/month** |
| Current vendor bot licence | Rs 3.2 lakh/year (Rs 26,667/month) |
| Hosting / infrastructure | Not included in the direct cost; not estimated. It runs on any machine with Python 3.11+, with no GPU. |
| Latency | ~18 ms per request including the explanation |
| Model size | 205 KB |

## Key business finding

The client asked us to match `team_label` at 90% or better, and the model does (96.8%). But
`team_label` agrees with the team that finally resolved the request (`final_team`) only **77.2%**
of the time, so reproducing the bot also reproduces its misroutes. The delivered model's agreement
with `final_team` is about the same as the bot's (77.1% against 76.7% in Apr–Jun 2026). We claim
the licence saving only, not fewer misroutes.

An analysis-only model trained on `final_team` reached 85.0% agreement with `final_team`. That is
an estimated ~166 → ~107 misroutes/month, about Rs 41k/month at policy §4 rates. It would match
`team_label` only about 79% of the time, so changing the target is a client decision. We recommend
a 2–4 week shadow run before switching off the vendor bot.

## How to run

```bash
git clone https://github.com/SANDEEP-K15/intelligent-routing.git
cd intelligent-routing
pip install -r requirements.txt
# copy router.joblib from the Drive folder into models/
python -m scripts.smoke_test
uvicorn service.app:app --port 8000
```

Then open http://localhost:8000. The API is `POST /api/route`. To regenerate everything from the
assignment data, run `python -m scripts.evaluate`, `python -m scripts.train` and
`python -m scripts.predict`.

## Deliverables in the project

- `README.md`: engineering documentation
- `docs/DECISIONS.md`: target and design decisions
- `docs/evaluation-report.md`: evidence (splits, baselines, metrics, confusion matrix, error analysis, cost)
- `docs/memo-to-ritu.md`: one-page client memo
- `docs/data-quality.md`, `docs/operations.md`, `docs/recording-checklist.md`: data findings, operational handoff, demo script
- `predictions.csv` and `models/router.joblib`: shared privately via Drive, never committed
- `kestrel_router/`, `scripts/`, `service/`, `tests/`: source code and tests
