# Evaluation report: Kestrel service-request routing

**Target:** `team_label`, the vendor bot's queue at creation, with the 15 Jan 2026 renames
normalised to the current names (see [DECISIONS.md](DECISIONS.md)).

**Primary metric:** exact-match accuracy, the share of requests sent to the same team as the
label. This is what "90% match" means.

**Secondary metrics:** macro-F1, per-team precision, recall and F1, and the confusion matrix.
Agreement with `final_team` is reported separately as the business view.

All numbers come from `python -m scripts.evaluate` (`reports/metrics.json`,
`reports/evaluation_tables.md`). `test_unlabelled.csv` was not used for any decision.

## 1. Dataset split

| Fold | Train | Validate | Train rows | Val rows |
|---|---|---|---|---|
| Primary | 1 Apr 2025 – 31 Mar 2026 | 1 Apr – 30 Jun 2026 | 8,687 | 2,135 |
| Backtest | 1 Apr 2025 – 31 Dec 2025 | 1 Jan – 31 Mar 2026 | 6,519 | 2,168 |
| Final fit | 1 Apr 2025 – 30 Jun 2026 | hidden test: 1 Jul – 30 Sep 2026 | 10,822 | 2,178 |

Each validation window is three months, immediately after its training period, which mirrors the
hidden test.

## 2. Baselines and candidate models

Values are accuracy, with macro-F1 in brackets.

| Model | Inputs | Apr–Jun 2026 | Jan–Mar 2026 | Mean |
|---|---|---|---|---|
| Majority class (always Repairs) | none | 0.289 (0.064) | 0.286 (0.064) | 0.288 |
| Rules / keywords (teams.csv wording + 3 known bot habits) | text, product | 0.910 (0.911) | 0.917 (0.917) | 0.914 |
| Logistic regression, word TF-IDF | text | 0.941 (0.942) | 0.949 (0.949) | 0.945 |
| Logistic regression, word + char TF-IDF | text | 0.950 (0.950) | 0.957 (0.956) | 0.954 |
| ↳ + channel, warranty | text, channel, warranty | 0.951 (0.951) | 0.953 (0.953) | 0.952 |
| ↳ + product_family | text, product | 0.958 (0.956) | 0.968 (0.965) | 0.963 |
| ↳ + product, balanced class weights | text, product | 0.960 (0.958) | 0.965 (0.963) | 0.963 |
| ↳ + all three fields | text, channel, warranty, product | 0.959 (0.957) | 0.967 (0.964) | 0.963 |
| LinearSVC (no confidence output) | text, product | 0.965 (0.964) | 0.971 (0.968) | 0.968 |
| **LinearSVC + sigmoid calibration (selected)** | **text, product** | **0.968 (0.967)** | **0.971 (0.968)** | **0.969** |
| ↳ + channel, warranty | text, channel, warranty, product | 0.967 (0.965) | 0.971 (0.968) | 0.969 |

The C values tried for logistic regression (1, 3, 10, 30) are in `reports/metrics.json`.

What the comparison shows:
- **The 90% bar is already met by hand-written rules** (91.4%). It can be met because the bot's
  behaviour is largely deterministic. The ML models add about 5.5 points over the rules and need
  no hand maintenance.
- **`product_family` genuinely helps:** +0.7 to +1.2 points across models and both folds. The
  bot uses the product field to send water-purifier faults and vague purifier requests to Filters &
  Consumables. In 16% of rows the field disagrees with the text, so it is kept as a separate input
  rather than replacing the text.
- **`channel` and `warranty_status` do not help** (−0.05 to −0.15 points of mean accuracy) and are
  excluded.
- **Class weighting changes nothing material.** Classes are only mildly imbalanced (9–29%).
- **Why the calibrated SVM:** it is the most accurate model that also gives a usable confidence.
  It beats logistic regression on the same inputs by 0.6 points on average, which is more than
  the 0.3-point simplicity tolerance. Both are linear models of the same size and speed.

## 3. Selected model: validation metrics

| | Apr–Jun 2026 (primary) | Jan–Mar 2026 (backtest) |
|---|---|---|
| Accuracy | **0.968** | **0.970** |
| 95% bootstrap CI | 0.961 – 0.976 | 0.963 – 0.977 |
| Macro-F1 | 0.967 | 0.968 |
| Monthly accuracy | Apr 0.979 · May 0.962 · Jun 0.965 | Jan 0.970 · Feb 0.964 · Mar 0.977 |

### Per team (primary fold)

| Team | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Repairs | 0.983 | 0.963 | 0.973 | 617 |
| Billing | 0.976 | 0.985 | 0.980 | 331 |
| Product Advice | 0.968 | 0.968 | 0.968 | 221 |
| Returns & Replacement | 0.980 | 0.956 | 0.968 | 252 |
| Warranty Claims | 0.946 | 0.965 | 0.955 | 199 |
| Filters & Consumables | 0.942 | 0.961 | 0.951 | 285 |
| Installs & Demo | 0.958 | 0.983 | 0.970 | 230 |

Every team scores ≥ 0.94 on precision and recall in both folds.

### Confusion matrix (primary fold; rows = true `team_label`, columns = prediction)

| | Repairs | Billing | Prod. Advice | Returns | Warranty | Filters & Cons. | Installs & Demo |
|---|---|---|---|---|---|---|---|
| **Repairs** | 594 | 5 | 2 | 1 | 4 | 10 | 1 |
| **Billing** | 0 | 326 | 0 | 1 | 1 | 2 | 1 |
| **Product Advice** | 2 | 2 | 214 | 0 | 2 | 0 | 1 |
| **Returns & Repl.** | 1 | 0 | 2 | 241 | 2 | 3 | 3 |
| **Warranty Claims** | 0 | 0 | 1 | 2 | 192 | 2 | 2 |
| **Filters & Cons.** | 5 | 1 | 1 | 0 | 2 | 274 | 2 |
| **Installs & Demo** | 2 | 0 | 1 | 1 | 0 | 0 | 226 |

The largest single confusion is Repairs ↔ Filters & Consumables (15 rows). Of these, 12 are
Water Purifier requests and 11 are water-purifier faults.

## 4. Error analysis (primary fold, 68 errors out of 2,135)

| Slice | Rows | Accuracy | Errors |
|---|---|---|---|
| Clear single-issue wording | 1,614 | 0.971 | 46 |
| Water-purifier fault | 82 | **0.817** | 15 |
| Mentions a payment ("paid", "payment done") | 127 | 0.953 | 6 |
| Vague ("someone contact me", "issue with…") | 312 | 0.997 | 1 |
| Several issues in one message | 283 | **0.915** | 24 |
| Single issue | 1,852 | 0.976 | 44 |

By channel, accuracy ranges from 0.943 (email) to 0.974. By warranty status it ranges from 0.963
to 0.976, and by product from 0.941 (Water Purifier) to 0.985. Water Purifier accounts for 38% of
errors.

Why the hard cases fail:

1. **Inconsistent labels (37 of 68 errors).** The model predicted the team the bot most often
   gave to that exact wording in training, but this request's label was the minority choice.
   Examples (product names shown as `<p>`):
   - "`<p>` making loud noise", labelled Installs & Demo
   - "`<p>` making loud noise", labelled Returns & Replacement
   - "best settings for `<p>`", labelled Repairs

   This is about 2–3% bot noise. No model can recover it from the request, and it sets a practical
   ceiling of about 97–98%.
2. **Water-purifier faults.** In training the bot sent purifier faults ending "…purifier not
   working" to Filters & Consumables 92% of the time. Without that phrase, it split them 55%
   Repairs and 39% Filters & Consumables: close to a coin toss, which the model cannot predict.
3. **Several issues in one message** (for example "`<p>` arrived damaged, need replacement",
   labelled Billing or Warranty Claims). The bot's choice among the issues is not consistent, and
   many of these combinations were never seen in training (22 of the 68 errors are on wording not
   seen in training).
4. **Payment mentions on vague requests** (for example "help `<p>` payment done", labelled
   Repairs, predicted Billing). The bot usually sends "paid" to Billing, but not always when the
   rest of the message is vague.

Row-level error listings are kept in `reports/private/validation_errors.csv` (git-ignored). They
are not reproduced here, because they contain customer messages.

## 5. Expected hidden-test score (Jul–Sep 2026, `team_label`)

| Evidence | Value |
|---|---|
| Primary-fold accuracy | 0.968 (CI 0.961–0.976) |
| Backtest accuracy | 0.970 (CI 0.963–0.977) |
| Range of the six validation months | 0.962 – 0.979 |
| Practical ceiling from label noise | ≈ 0.97–0.98 |
| Test rows whose wording was seen in training (discovery) | 93.4% (validation fold: 92.7%) |

**Expected hidden-test accuracy: 95.5% – 97.5%, most likely about 96.5%.** This is not
guaranteed. It assumes the bot routed Jul–Sep 2026 the way it did during the previous 15 months.

Reasons the estimate is credible:
- bot-versus-outcome agreement was flat at 74–79% in every month;
- the mix of request patterns in the test file matches training to within 0.3 points;
- the test file is all-CRM, so there is no encoding corruption;
- the renames are already handled.

The final model is trained on 3 more months than the primary fold, which should help slightly.
The main downside risk is a change in the bot's own rules after June 2026. That cannot be detected
from the unlabelled test file.

## 6. Comparison with the existing bot: contract vs outcome

`final_team` is the team that closed the request (resolution log). It is used **only for
evaluation**.

| | Apr–Jun 2026 | Jan–Mar 2026 | All history |
|---|---|---|---|
| **A.** Our model matches `team_label` | **0.968** | **0.970** | |
| **B.** Bot `team_label` matches `final_team` | 0.767 | 0.775 | **0.772** |
| **B.** Our model matches `final_team` | 0.771 | 0.784 | |
| Analysis only: same model trained on `final_team`, matched against `final_team` | 0.850 | 0.852 | |

### Agreement with `final_team` by request pattern (Apr–Jun 2026)

| Pattern | Bot label | Our model | Final-team model (analysis only) |
|---|---|---|---|
| Clear | 0.959 | 0.967 | 0.968 |
| Mentions payment | 0.039 | 0.039 | 0.850 |
| Water-purifier fault | 0.293 | 0.256 | 0.976 |
| Vague | 0.196 | 0.196 | 0.205 |

What this means:
- **Reproducing `team_label` reproduces the bot's misroutes.** Our model and the bot agree with the
  final team equally often (about 77%).
- **No reduction in misroutes is claimed for the delivered model.**
- **A model trained on `final_team` would fix most payment-mention and purifier-fault misroutes.**
  It would not fix vague requests: about 15% of traffic, where only asking the customer helps.
- **That model would score only about 79% against `team_label`,** so it fails the client's stated
  90% contract. Changing the target is a client decision.

## 7. Cost

### Direct model cost

| Item | Value |
|---|---|
| Direct model/API cost per prediction | **Rs 0** (local CPU inference, no paid API) |
| Monthly at ~700 requests | 700 × Rs 0 = **Rs 0/month** |
| Vendor bot licence | Rs 3.2 lakh/year = Rs 26,667/month |
| Hosting | Not included. It runs on any existing machine with Python; no infrastructure cost was estimated. |
| Latency | ~18 ms per single request including explanation; ~0.25 ms per request in batch |
| Model artefact | 205 KB |

### Misroute cost at ops policy §4 rates

Each misrouted request costs Rs 305 per transfer plus Rs 260 for the extra contact. Historically
there were 1.44 transfers per misrouted request, so each misroute costs about Rs 698.

| Apr–Jun 2026, per month | Misrouted requests | Estimated misroute cost |
|---|---|---|
| Bot labels | 166 | Rs 1,15,878 |
| Our model (`team_label`) | 163 | Rs 1,13,551 |
| Final-team model (analysis only, not delivered) | 107 | Rs 74,692 |

These are estimates. Counterfactual transfer counts for a different routing decision are not
observed. The small gap between the bot and our model is within noise and is **not** claimed as a
saving.
