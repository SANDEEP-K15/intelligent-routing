# Screen-recording checklist (aim for 5–7 minutes)

## Before recording

- [ ] `models/router.joblib` and `predictions.csv` exist locally. If not, run
      `python -m scripts.train` and then `python -m scripts.predict`.
- [ ] `python -m scripts.release_check` shows PASS for every step.
- [ ] The service is running: `uvicorn service.app:app --port 8000`
- [ ] Close any window showing raw assignment data. Do not open `train.csv` or
      `test_unlabelled.csv` on screen.

## Suggested running order

1. **The problem and the decision (30 s).**
   - The client wants a 90% match with the bot's labels.
   - The licence is Rs 3.2 lakh/year.
   - The labels agree with the closing team only 77.2% of the time.
2. **Repository tour (45 s).**
   - The `kestrel_router/` package, `scripts/`, `service/`, `tests/` and `docs/`.
   - Point out `.gitignore` and `scripts/audit_public_repo.py`: no data in the public repo.
3. **Validation evidence (90 s).**
   - In `docs/evaluation-report.md`, show the baselines (28.8% majority, 91.4% rules) and the
     selected model (96.8% Apr–Jun, 97.0% Jan–Mar).
   - Show the confusion matrix and the error analysis.
4. **The contract vs outcome table (60 s).**
   - The bot vs the closing team: 76.7%; our model: 77.1%.
   - A final-team model would reach 85.0%. Why that is the client's decision.
5. **Live UI (60 s).** At http://localhost:8000, route three requests:
   - a clear fault (expect Repairs, high confidence);
   - a message with "…I paid extra for this" (Billing, with the policy §3 note);
   - "someone please contact me about my fan" (Repairs, with the vague-request behaviour).
6. **API (30 s).** Run this in a terminal and show the JSON:
   `curl -X POST localhost:8000/api/route -H "Content-Type: application/json" -d @examples/route_request.json`
7. **Tests and checks (30 s).** Run `python -m scripts.release_check` and show the summary.
8. **Cost and next steps (30 s).**
   - Rs 0/month model cost.
   - A 2–4 week shadow run before switching off the bot.
