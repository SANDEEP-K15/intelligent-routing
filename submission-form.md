# Submission form: Task 2 V2, Kestrel Home service-request routing

> Three fields are marked **UPDATE BEFORE SUBMISSION** because the links or values do not exist yet.
> Every other field is final.

## Links

| Field | Value |
|---|---|
| GitHub repository | https://github.com/SANDEEP-K15/intelligent-routing  |
| Google Drive folder | https://drive.google.com/drive/folders/1SLMLNTJsiBRlZ8QQ_NciiOWvHQPN8zDp?usp=sharing |
| Screen recording | https://drive.google.com/file/d/1dmSefLiUIGlYK8wfo09YM3OIu_cVmlde/view?usp=sharing |

## Effort

| Field | Value |
|---|---|
| Hours spent | Around 6-7 Hours |

## What was built

A local service that suggests the team queue for a new Kestrel service request:

- **Model:** word and character TF-IDF over the cleaned request text, plus `product_family`,
  with a calibrated linear SVM (scikit-learn). It runs on a CPU and calls no AI or paid API.
- **Service:** a FastAPI endpoint (`POST /api/route`). It returns the team, a confidence, plain-language
  reasons and the runner-up teams. A one-page UI calls the same endpoint.
- **Batch predictions:** `predictions.csv` covers all 2,178 test requests and is validated before it
  is written.
- **Supporting pieces:**
  - a time-based evaluation pipeline;
  - 82 automated tests;
  - CI;
  - a public-repo data audit;
  - an operations runbook and a one-page memo for the client.

## Business decision (numbers and rupees)

**Replace the vendor bot after a 2–4 week shadow run.**

- **Accuracy:** the model matches the bot's routing labels **96.8%** of the time on the most recent
  held-out quarter (Apr–Jun 2026), against the client's 90% bar.
- **Cost:** it runs at **Rs 0/month** direct model cost, instead of the **Rs 3.2 lakh/year
  (Rs 26,667/month)** licence.

**Caveat:** the bot's labels agree with the team that finally closed the request only **77.2%** of
the time. A model that copies the labels copies the misroutes too:

- about **166 misroutes a month**, roughly **Rs 1.16 lakh/month** at policy §4 rates (estimate);
- the only saving claimed is the licence.

## Expected hidden-test score and metric

- **Metric:** exact-match accuracy against `team_label`, the bot's queue at creation, using the seven
  current team names.
- **Expected:** **95.5% – 97.5%, most likely about 96.5%.** This is an estimate, not a guarantee. It
  assumes the bot routed Jul–Sep 2026 the way it did in the previous 15 months.

| Evidence | Value |
|---|---|
| Validation accuracy, Apr–Jun 2026 | 96.8% (2,067 / 2,135; 95% CI 96.1–97.6%), macro-F1 0.967 |
| Backtest accuracy, Jan–Mar 2026 | 97.0% (2,104 / 2,168; 95% CI 96.3–97.7%), macro-F1 0.968 |
| Monthly range across the six validation months | 96.2% – 97.8% |
| Baselines (mean of both folds) | Majority class 28.8%; keyword rules 91.4%; logistic regression 96.3% |

## Validation methodology

- **Time-based, not random.**
  - Primary: train Apr 2025 – Mar 2026, validate Apr–Jun 2026.
  - Backtest: train Apr–Dec 2025, validate Jan–Mar 2026.
- **The test file was never used** for any decision.
- **Model selection:** the simplest of 15 candidate models within 0.3 points of the best mean
  accuracy over both folds. Only models that output a confidence were eligible.
- **Final fit:** the selected model is refit on all 10,822 labelled requests.
- **No leakage:** resolution-log fields (`first_team`, `final_team`, `transfers`, `resolved_at`),
  `team_label`, `request_id` and timestamps are never model inputs. This is enforced in code and
  tests.
- **Outcome evaluation:** `final_team` is used only to measure how the labels and predictions
  compare with actual outcomes.

## Where the model is wrong

There were 68 errors out of 2,135 in Apr–Jun 2026:

- **Inconsistent bot labels (37 of 68):** identical wording carries different bot labels. This is
  about 2–3% noise, which caps achievable accuracy at about 97–98%.
- **Water-purifier faults: 81.7% accuracy.** Without the phrase "…purifier not working", the bot
  itself split these 54% Repairs and 39% Filters & Consumables.
- **Messages with several issues: 91.5% accuracy.** 22 of the 68 errors are on wording not seen in
  training.
- **Payment mentions in vague messages:** the bot does not always send these to Billing.
- **Against actual outcomes:** vague "please call me" requests (about 15% of traffic) reach the
  right team about 20% of the time with the bot or our model. Only asking the customer fixes these.

## Changes and pushback

- **The labels are not ground truth.**
  - The client called the labels "ground truth". We showed they agree with the closing team only
    77.2% of the time.
  - We kept `team_label` as the contract target, as briefed, but report both measures and do not
    claim reduced misroutes.
- **Shadow before switch-off.** We recommend a 2–4 week shadow run rather than switching the bot
  off immediately.
- **A possible change of target.** We put a decision to Ritu: aim for "right team first time"
  instead. An analysis-only model trained on the closing team reached 85.0% agreement with it.
  That is an estimated ~166 → ~107 misroutes/month, about Rs 41,000/month. It would match the bot's
  labels only about 79% of the time, so it fails the current 90% contract.
- **Headcount.** We advised using closing-team volumes for headcount planning, not bot queue counts.

## Data and handoff issues

- **Renames.** Two teams were renamed on 15 Jan 2026. Labels are normalised to the current names,
  and the output uses only those names.
- **Legacy Zoho records:**
  - 480 texts had encoding corruption, now repaired;
  - resolution times were stored in UTC (1,140 apparent negative durations). They are not used as
    features.
- **Transfer counts** contradict the first and final team in 477 rows.
- **`product_family`** disagrees with the product named in the text in about 16% of rows.
- **Coverage:** the data covers 15 months, not the 18 mentioned in the email. The resolution log
  covers every training request.
- **Model file.** Policy §10 forbids publishing the data, so `predictions.csv` and the model
  artefact are shared privately. A fresh clone needs `router.joblib` copied into `models/`.

## Deliberately not built

- **Not built at all:**
  - no LLM or paid API;
  - no RAG, vector database, LangChain/LangGraph or agents;
  - no cloud deployment, database or microservices;
  - no authentication;
  - no complex frontend.
- **Documented as a runbook rather than automated:** monitoring and retraining
  (`docs/operations.md`).
- **Analysis only, not used for `predictions.csv`:** the final-team model.

## Unexpected findings

- **The bot is systematically wrong in three places** (agreement with the closing team,
  Apr–Jun 2026):
  - requests mentioning a payment go to Billing (about 4% correct);
  - water-purifier faults go to Filters & Consumables (about 29% correct);
  - vague requests go to Repairs (about 20% correct).
- **Hand-written keyword rules already reach 91.4%,** because the bot is largely deterministic.
- **`product_family` improves accuracy by 0.7–1.2 points,** because the bot reads that field, even
  though it conflicts with the text in about 16% of rows.
- **Messages are built from about 970 recurring phrases.** 93.4% of test wording appears in
  training.
- **Bot queue counts misstate workload.** For example, the Repairs queue receives ~206 requests a
  month, but Repairs closes ~165.

## AI tools used

- **During development:** Claude Code (Anthropic's Claude Opus 5.5) was used as a coding assistant
  for data exploration, implementation, tests, documentation and the git workflow. Every reported
  figure was produced by running the scripts in this repository.
- **At prediction time:** no AI service is called.

## Costs

| Item | Value |
|---|---|
| Direct model/API cost per prediction | Rs 0 (local CPU inference) |
| Monthly model/API cost at ~700 requests/month | 700 × Rs 0 = **Rs 0/month** |
| Vendor bot licence being replaced | Rs 3.2 lakh/year (Rs 26,667/month) |
| Hosting / infrastructure | Not included in the direct cost and not estimated. It runs on any machine with Python 3.11+, with no GPU. |
| Latency | ~18 ms per request including the explanation |
| Model size | 205 KB |

## Approaches tried and discarded

| Approach | Why it was discarded |
|---|---|
| LLM / embedding routing | Per-request cost or a paid key, large downloads, and it would not fix the vague requests that limit accuracy |
| Keyword rules only | 91.4%; brittle to new wording |
| Logistic regression | 96.3% vs 96.9% mean accuracy for the calibrated SVM |
| Raw LinearSVC | No confidence output |
| Word-only TF-IDF | 94.5% |
| Balanced class weights | No material change |
| Channel / warranty features | −0.02 to −0.14 points |
| Random train/test split | Rejected in favour of time-based validation |
| Training `predictions.csv` on `final_team` | Fails the 90% label contract (about 79%); kept as analysis only |

## Public repository

https://github.com/SANDEEP-K15/intelligent-routing

- **Contains:** code, tests, documentation, aggregate metrics, and examples with fictitious IDs.
- **Excludes:** assignment data, `predictions.csv`, the model artefact and row-level reports.
- **Guarded by:** `scripts/audit_public_repo.py` and CI (lint, audit and tests on Python
  3.11–3.13).

## Monday handoff

1. Copy `router.joblib` from the private Drive folder into `models/`, then run
   `python -m scripts.smoke_test`.
2. Start the service: `python -m uvicorn service.app:app --port 8000`
3. Start the shadow run. Call `/api/route` for every new request and log it next to the bot's queue.
   Nothing changes for agents.
4. Each week, check:
   - agreement with the bot (target ≥ 90%);
   - agreement with the closing team as requests close;
   - the share of low-confidence predictions.
5. After 2–4 weeks, Ritu decides on switching off the bot and on whether to retarget to the
   closing team.
6. The runbook (monitoring thresholds, retraining) is in `docs/operations.md`.

## How to run

```bash
git clone https://github.com/SANDEEP-K15/intelligent-routing.git
cd intelligent-routing
pip install -r requirements.txt
# copy router.joblib from the Drive folder into models/
python -m scripts.smoke_test
python -m uvicorn service.app:app --port 8000
```

Then open http://localhost:8000. To regenerate everything from the assignment data, run
`python -m scripts.evaluate`, `python -m scripts.train` and `python -m scripts.predict`.

## Deliverables in the project

- `README.md`: engineering documentation
- `docs/DECISIONS.md`: target and design decisions
- `docs/evaluation-report.md`: evidence (splits, baselines, metrics, confusion matrix, error analysis, cost)
- `docs/memo-to-ritu.md`: one-page client memo
- `docs/data-quality.md`, `docs/operations.md`, `docs/recording-checklist.md`: data findings,
  operational handoff, demo script
- `predictions.csv` and `models/router.joblib`: shared privately via Drive, never committed
- `kestrel_router/`, `scripts/`, `service/`, `tests/`: source code and tests
