# Kestrel Home: service-request routing

[![CI](https://github.com/SANDEEP-K15/intelligent-routing/actions/workflows/ci.yml/badge.svg)](https://github.com/SANDEEP-K15/intelligent-routing/actions/workflows/ci.yml)

This is a local, CPU-only classifier that suggests the team queue for a new Kestrel Home service
request. It is built to replace the vendor routing bot (licence Rs 3.2 lakh/year) and uses no
paid model API.

| | |
|---|---|
| Target | `team_label`: the queue the existing bot assigned at creation (client requirement: ≥ 90% match) |
| Model | Word and character TF-IDF over the cleaned request text, plus `product_family`, with a calibrated linear SVM (scikit-learn) |
| Validation (time-based) | **96.8%** accuracy on Apr–Jun 2026; **97.0%** on Jan–Mar 2026 |
| Expected hidden-test accuracy | 95.5% – 97.5% (not guaranteed) |
| Direct model/API cost | Rs 0 per prediction, i.e. Rs 0/month at ~700 requests/month |
| Main caveat | `team_label` agrees with the team that finally resolved the request only **77.2%** of the time |

| Document | Contents |
|---|---|
| [docs/DECISIONS.md](docs/DECISIONS.md) | Target choice, team naming, validation design, no paid API, data policy |
| [docs/evaluation-report.md](docs/evaluation-report.md) | Splits, baselines, metrics, confusion matrix, error analysis, expected score, cost |
| [docs/data-quality.md](docs/data-quality.md) | Data-quality findings and how the pipeline handles them |
| [docs/operations.md](docs/operations.md) | Deployment, shadow rollout, monitoring, retraining |
| [docs/memo-to-ritu.md](docs/memo-to-ritu.md) | One-page decision memo for the client |
| [submission-form.md](submission-form.md) | Assignment submission summary |

---

## 1. Problem

Kestrel's D2C service desk receives requests over IVR, chat, WhatsApp and email. A vendor bot
assigns each one to one of seven team queues when it is created. Agents transfer requests that
reach the wrong team. The client asked for a classifier that matches the bot's historical labels
at 90% or better, so that the bot can be retired.

## 2. Business context: the contract and the outcome

The data supports two separate questions, and this project keeps them apart.

| Question | Measure | Result (Apr–Jun 2026) |
|---|---|---|
| **A. Contract.** Do we reproduce the routing labels the client asked for? | model vs `team_label` | **96.8%** |
| **B. Outcome.** Do those labels send requests to the team that finally resolves them? | bot `team_label` vs `final_team` | **76.7%** (77.2% over all history) |
| | our model vs `final_team` | 77.1% |

The model reproduces the bot, including its systematic misroutes:
- requests that mention a payment are sent to Billing;
- water-purifier faults are sent to Filters & Consumables;
- vague "please call me" requests are sent to Repairs.

Replacing the bot with this model saves the licence. It does **not** reduce transfers.

A separate analysis-only model trained on `final_team` reached 85.0% agreement with
`final_team`. That suggests misroutes could fall by roughly a third if the client chose to change
the target. This is not what `predictions.csv` contains (see D1 in
[docs/DECISIONS.md](docs/DECISIONS.md)).

## 3. Architecture

```
original CSVs (read-only)
   │  kestrel_router/data.py     schema validation, rename normalisation, fail loudly
   │  kestrel_router/text.py     Zoho mojibake repair, accent folding, ID removal, lowercasing
   ▼
cleaned frame ──► scripts/evaluate.py   time-based folds, model comparison, error analysis
               ──► scripts/train.py      fit selected spec on all labelled data → models/router.joblib
               ──► scripts/predict.py    predictions.csv (validated before it is written)
                                         │
models/router.joblib ──► service/app.py (FastAPI) ──► POST /api/route   (JSON in, team + reasons out)
                                                   └─► GET  /         (single HTML page)
```

There is no database, vector store, LLM, agent framework, cloud dependency or background worker.

```
kestrel_router/   config, data validation, text cleaning, model, metrics, rules baseline, submission checks
scripts/          evaluate · train · predict · validate_submission · smoke_test · audit_public_repo · release_check
service/          FastAPI app + static/index.html
tests/            pytest suite (synthetic fixtures only)
reports/          aggregate metrics (metrics.json, evaluation_tables.md, selected_spec.json)
docs/             decisions, evaluation report, data quality, operations, memo, recording checklist
examples/         API payload and predictions-file format with fictitious IDs
models/           README only; router.joblib is shared privately
```

## 4. Model

- **Inputs:**
  - `request_text`, cleaned;
  - `product_family`, kept because validation showed it adds 0.7–1.2 points (see below).
- **Inputs tested and excluded:** `channel` and `warranty_status`. They add nothing:
  - logistic regression, text only: 95.4% mean accuracy → 95.2% with channel and warranty added;
  - calibrated SVM: 96.9% with product → 96.9% with all three fields.
- **Never used (enforced in code and tests):** `first_team`, `final_team`, `transfers`,
  `resolved_at`, `team_label`, `request_id`, `created_at_ist` and `source`.
- **Features:**
  - word TF-IDF (1–2 grams, min_df 2, sublinear tf);
  - character TF-IDF (`char_wb`, 2–5 grams, min_df 3), which absorbs spelling noise;
  - one-hot `product_family`, with unknown values ignored.
- **Classifier:** `LinearSVC(C=1)` wrapped in `CalibratedClassifierCV(method="sigmoid", cv=5,
  ensemble=False)`. One linear model is trained on all rows, and calibration supplies the
  confidence score.
- **Selection rule:** the simplest candidate within 0.3 points of the best mean accuracy over both
  time folds. Only models that output a confidence were eligible.

  | Model | Mean accuracy |
  |---|---|
  | Calibrated SVM | 96.9% |
  | Logistic regression (same inputs) | 96.3% |
  | Rules baseline | 91.4% |
  | Majority baseline | 28.8% |

- **Labels:** "Installations" is normalised to "Installs & Demo" and "Consumables" to "Filters &
  Consumables" (rename of 15 Jan 2026, ops policy §5). Output is always one of the seven current
  names.

### Why `product_family` helps

The bot sends a water purifier's fault or vague request to Filters & Consumables based on the
product field, even when the text names another product. In 16% of rows the field and the text
disagree, so the field is kept as a separate input rather than trusted over the text.

## 5. Validation

| Fold | Train | Validate | Rows (train / val) | Accuracy | Macro-F1 |
|---|---|---|---|---|---|
| Primary | 1 Apr 2025 – 31 Mar 2026 | 1 Apr – 30 Jun 2026 | 8,687 / 2,135 | **0.968** (95% CI 0.961–0.976) | 0.967 |
| Backtest | 1 Apr 2025 – 31 Dec 2025 | 1 Jan – 31 Mar 2026 | 6,519 / 2,168 | **0.970** (95% CI 0.963–0.977) | 0.968 |

- The test file was never used for tuning.
- The final model is refit on all 10,822 labelled rows.
- Monthly accuracy stayed between 96.2% and 97.9% across all six validation months.

Full metrics, the confusion matrix and the error analysis are in
[docs/evaluation-report.md](docs/evaluation-report.md). The generated tables are in
`reports/evaluation_tables.md` and `reports/metrics.json`.

## 6. Cost

| Item | Monthly |
|---|---|
| Vendor bot licence (Rs 3.2 lakh/year) | Rs 26,667 |
| This model: direct model/API cost (700 requests × Rs 0) | **Rs 0** |
| Hosting / infrastructure | Not estimated. It runs on any machine that runs Python and needs no GPU. Existing Kestrel infrastructure costs are outside the direct prediction cost. |

Measured on a laptop CPU:
- about 10–30 ms for one request including the explanation;
- under 0.3 ms per request in batch;
- the model artefact is about 205 KB.

## 7. Setup

Python 3.11 or newer.

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS / Linux
pip install -r requirements.txt
```

**Model artefact.** `models/router.joblib` is not in the repository, because its vocabulary is
derived from customer text (see [models/README.md](models/README.md)). Copy the privately shared
`router.joblib` into `models/`, then check the installation:

```bash
python -m scripts.smoke_test
```

### Reproducing everything from the assignment data

Place the eight assignment files in the project root; `.gitignore` keeps them out of git. Then:

```bash
python -m scripts.evaluate      # time-based model comparison; writes reports/ (about 5 min)
python -m scripts.train         # fits the selected model on all 10,822 labelled requests
python -m scripts.predict       # writes predictions.csv, only if it passes validation
python -m scripts.validate_submission predictions.csv
```

- **Format:** `predictions.csv` has exactly the format of `sample_submission.csv`: header
  `request_id,team`, one row per test request in test-file order, and only the seven current team
  names. See [examples/predictions.example.csv](examples/predictions.example.csv), which uses
  fictitious IDs.
- **Not committed:** the real file is git-ignored, because it is request-level output.

## 8. API

Start the service:

```bash
uvicorn service.app:app --port 8000
```

`POST /api/route` (example payload in [examples/route_request.json](examples/route_request.json))

```json
{
  "request_text": "good morning, water purifier not turning on",
  "product_family": "Water Purifier",
  "channel": "chat",
  "warranty_status": "in_warranty"
}
```

- `request_text` is required: 1–2,000 characters, not blank.
- The other fields are optional. Values outside the documented lists are rejected with 422.
- `channel` and `warranty_status` are accepted for compatibility but are not used by the current
  model. The `inputs_used` field in the response says which inputs were used.

Response:

```json
{
  "team": "Filters & Consumables",
  "confidence": 0.958,
  "confidence_band": "high",
  "reasons": [
    "The request mentions \"not turning\", wording that is typically routed to Filters & Consumables.",
    "The product family is 'Water Purifier', which historically leans towards Filters & Consumables.",
    "Note: historical routing sends water-purifier faults to Filters & Consumables; most of these were finally resolved by Repairs."
  ],
  "alternatives": [{"team": "Repairs", "probability": 0.027}, {"team": "Product Advice", "probability": 0.01}],
  "inputs_used": ["request_text", "product_family"],
  "model_version": "…",
  "latency_ms": 19.5
}
```

- **Reasons** quote the phrases in the request that pushed the decision. They also flag the three
  known historical routing habits, and low confidence (below 0.5).
- `GET /api/health` reports the model status and validation metrics.
- If the artefact is missing, the service still starts. `/api/route` then returns 503 with
  instructions.

## 9. UI

`GET /` serves one static HTML page (`service/static/index.html`). On it you can:
- type a message;
- optionally choose the product, the only optional field the model uses (the page reads this from
  `/api/health`);
- submit, and see the team, confidence, reasons and runner-up teams.

It has no build step and no framework.

## 10. Tests and checks

```bash
pip install -r requirements-dev.txt
pytest                               # synthetic fixtures; assignment-data tests skip when data is absent
ruff check .
python -m scripts.audit_public_repo  # blocks restricted data or secrets in the git index
python -m scripts.release_check      # all of the above + smoke test + predictions validation
```

CI (GitHub Actions) runs lint, the data audit and the test suite on Python 3.11–3.13 for every
push.

The tests cover:
- text cleaning and Zoho mojibake repair;
- team-rename mapping;
- schema validation and clear failures;
- time splits;
- leakage, i.e. predictions do not change when post-routing columns change;
- model save and load, and unknown or missing categories;
- prediction schema;
- the API (happy path, invalid input, missing model) and the UI route;
- submission validation;
- the post-install smoke test.

## 11. Limitations

- **The target reproduces the bot.** About 23% of `team_label` values disagree with the team that
  finally resolved the request. A high match score is not the same as correct routing.
- **There is a label-noise ceiling.** About 2–3% of identical-looking requests carry different
  bot labels, so roughly 97–98% is the practical maximum against `team_label`.
- **Water-purifier faults are the weakest group** (82% on validation). The bot itself routes them
  inconsistently.
- **Requests with several issues** score 91.5%. The model picks one team; the bot's choice among
  the issues is not always consistent.
- **Vague requests** are matched easily against the bot (it defaults them to Repairs). Neither the
  bot nor any text model can reliably find their real owner without asking the customer.
- **Drift is not monitored automatically.** Accuracy should be re-checked monthly against closed
  requests.
- **The hidden-test estimate assumes the bot behaved in Jul–Sep 2026 as it did before.**

## 12. Data handling

Ops policy §10: customer and operational data "must not be published, uploaded to public
repositories or shared beyond the engagement team".

- `.gitignore` excludes:
  - every assignment file;
  - all `*.csv` and `*.pdf` files;
  - `predictions.csv`;
  - `reports/private/` (row-level error listings);
  - `models/*.joblib`.
- **Committed documentation contains only aggregates** and templated wording, with product names
  replaced by `<p>`. It contains no customer messages or request IDs.
- **`scripts/audit_public_repo.py` blocks a push** if the index contains any of the following:
  restricted files, real request or order numbers, likely secrets, or (when the data is present
  locally) verbatim customer text. CI runs the same audit.
- **The pipeline never modifies the source files.** It opens them read-only and writes only to
  `models/`, `reports/` and `predictions.csv`.
