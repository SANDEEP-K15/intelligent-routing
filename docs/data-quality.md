# Data-quality findings

These findings come from the discovery analysis of the assignment pack. All figures are
aggregates; no customer text or request IDs are reproduced. Where the pipeline handles a finding,
the relevant code is named.

## Files

| File | Rows | Notes |
|---|---|---|
| `train.csv` | 10,822 | 1 Apr 2025 – 30 Jun 2026. IDs contiguous and time-ordered. No empty cells, malformed rows or duplicate IDs. |
| `test_unlabelled.csv` | 2,178 | 1 Jul – 30 Sep 2026. Strictly after training, all from the CRM. |
| `resolution_log.csv` | 10,822 | Exactly one row per training request. Nothing for test requests. |

## Findings

| # | Finding | Size | Handling |
|---|---|---|---|
| 1 | **Team renames.** "Installations" became "Installs & Demo" and "Consumables" became "Filters & Consumables" on 15 Jan 2026. The switch-over is exact, with no stray names on either side. | 9 label values for 7 teams | Normalised to current names (`data.normalize_team`). Validation fails if an old name appears after the rename date. |
| 2 | **Zoho text corruption.** UTF-8 text decoded as Windows-1252 (e.g. `urgÃ©nt`, `â€¦`). | 480 of 4,320 Zoho rows (11.1%); no CRM or test rows | Repaired by re-encoding (`text.repair_mojibake`); all 480 repair cleanly. |
| 3 | **Zoho resolution times stored in UTC**, not converted (ops policy §9). | 1,140 Zoho rows "resolve before they were created" | Adding 5h30 removes every negative duration. Only affects `resolved_at`, which is never a model input. |
| 4 | **`team_label` ≠ `final_team`.** The bot's queue differs from the team that closed the request. | 22.8% of training rows | Kept as two separate targets; see [DECISIONS.md](DECISIONS.md). |
| 5 | **Contradictory transfer counts.** | 126 rows with different first and final teams but 0 transfers; 351 rows with the same team but 1–2 transfers (possibly sent out and back) | Not used. Transfers feed only the cost estimate, as an average. |
| 6 | **`product_family` disagrees with the product named in the text.** | ~16% of rows that name a product, in both train and test | Kept as a separate input, never used to overwrite the text. Validation shows it still helps, because the bot reads it. |
| 7 | **Repeated wording.** Messages are built from about 970 recurring phrases. Some exact texts repeat, and identical texts sometimes carry different bot labels. | 663 duplicate training texts; ~2–3% label inconsistency | Duplicates kept (they are real requests). The inconsistency sets the practical accuracy ceiling (~97–98%). |
| 8 | **Order and registration numbers in text** do not link related requests: repeated order numbers point to different products. | ~2,100 order numbers, ~1,400 registration numbers | Removed during cleaning (`text.remove_identifiers`). No signal, and they identify customers. |
| 9 | **The resolution log covers every training request**, although the email expected the newest to be missing. | — | No action. The test period has no log, as expected. |

## Distribution stability

- **Team mix:** the share of each final team per quarter stays within about ±2 points.
- **Bot vs final team:** monthly agreement stays between 74% and 79%.
- **Request patterns:** the shares of clear, vague, payment-mention and purifier-fault requests in
  the test file match training to within 0.3 points.
- **Wording overlap:** 93.4% of test requests use wording seen in training.
