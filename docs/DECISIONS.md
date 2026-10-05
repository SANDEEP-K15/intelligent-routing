# Decision log

## D1 — Prediction target for `predictions.csv` is `team_label` (decided 6 Oct 2026)

**Decision.** The production model predicts `team_label`: the queue the vendor routing bot
assigned when the request was created. `final_team` (the team that closed the request, from
`resolution_log.csv`) is kept as a separate evaluation and business-analysis target. It is never
used as a model feature and never silently substituted for `team_label`.

**Reasoning.**

- The client brief is explicit. Ritu Deshpande: "Build a classifier that matches it at 90%+",
  "90% match is the bar", with the labels described as "eighteen months of real routing".
  Tanmay Kulkarni confirms that `team_label` "is the queue the bot put each request in when it
  came in".
- Discovery showed that `team_label` agrees with `final_team` for only **77.2%** of training
  requests (8,351 of 10,822). The disagreements are systematic, not random:
  - requests mentioning a payment are queued to Billing;
  - water-purifier faults are queued to Consumables;
  - vague "please call me" requests are queued to Repairs.

  Ops policy §3 says a payment mention does not make a request a Billing request.
- A model that reproduces `team_label` therefore reproduces the bot's misroutes as well. The
  project reports two separate questions and does not merge them:
  - **A. Contract:** how well do we reproduce the routing labels the client asked for?
  - **B. Outcome:** how do those labels, and our predictions, compare with the team that
    actually closed the request?
- No claim of reduced misroutes is made unless evidence on `final_team` supports it.

## D2 — Output uses the current seven team names

The test period (1 Jul – 30 Sep 2026) is after the 15 Jan 2026 rename (ops policy §5,
teams.csv). Historical labels are normalised as follows:

| Historical label | Current label |
|---|---|
| Installations | Installs & Demo |
| Consumables | Filters & Consumables |

Responsibilities did not change, so the merge loses no information. The seven output labels are:
Repairs, Billing, Product Advice, Returns & Replacement, Warranty Claims, Filters & Consumables,
Installs & Demo.

## D3 — Validation is time-based

The test set is the three months after training ends. Validation mirrors that layout.

| Fold | Train | Validate |
|---|---|---|
| Primary | 1 Apr 2025 – 31 Mar 2026 | 1 Apr – 30 Jun 2026 |
| Backtest | 1 Apr 2025 – 31 Dec 2025 | 1 Jan – 31 Mar 2026 |

The final model is refit on all labelled data. `test_unlabelled.csv` is never used for tuning.

## D4 — No paid model API

Routing is a local scikit-learn model on CPU. The direct model/API cost is Rs 0 per
prediction. This responds to Finance's requirement that the replacement must not be "an AI bill
that grows with every request".

## D5 — Source data is never published

Ops policy §10 forbids publishing the data or uploading it to public repositories. `.gitignore`
excludes all source files, all row-level derived outputs and the trained model artefact (its
vocabulary is derived from customer text). See README, "Data handling".
