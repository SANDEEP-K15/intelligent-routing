# Memo: replacing the vendor routing bot

**To:** Ritu Deshpande, Head of D2C Operations
**Cc:** Farhan Sheikh (Finance), Meenal Joshi (Service Desk), Tanmay Kulkarni (Data & IT)
**From:** Kabir Nanda's team
**Date:** 6 October 2026

## Decision

**Yes. The replacement meets your 90% bar, and it costs Rs 0 a month to run instead of Rs 3.2
lakh a year.**

- **Accuracy:** it routed requests to the same team as the bot **96.8%** of the time on the most
  recent three months it had not seen (Apr–Jun 2026), and 97.0% on the quarter before that.
- **Cost:** it runs on an ordinary computer and calls no paid AI service. The direct running cost
  is **Rs 0 per request, so Rs 0/month at ~700 requests**, against today's licence of **Rs 3.2
  lakh/year (Rs 26,667/month)**. Hosting it on existing Kestrel equipment is outside this figure.

**One important caveat.** The bot's labels are not the same as correct routing. Across 15 months,
the bot's first choice matched the team that finally closed the request only **77.2%** of the time.
A replacement that copies the bot also copies its misroutes: about **165 a month**, worth roughly
**Rs 1.15 lakh/month** in transfers and repeat contacts at policy rates. Switching saves the
licence; on its own it does not reduce transfers. We are not claiming any saving beyond the
licence.

## Where the bot goes wrong (Meenal's observations confirmed)

- **"I paid…" requests go to Billing.** Nearly all of them end up with another team. Policy §3 says
  a payment mention does not make it a billing issue.
- **Water-purifier breakdowns go to Filters & Consumables.** Most are closed by Repairs.
- **"Please call me about my product" goes to Repairs by default.** About 15% of requests say too
  little for anyone to route without asking.

A version trained on where requests *actually ended up* would fix most of the first two problems.
On the same test months it cut estimated misroutes from about 166 to 107 a month, roughly
**Rs 40,000/month** in handling cost. It would match the bot's labels only about 79% of the time,
so it would fail the 90% bar as currently written. That is your call, not ours. These figures are
estimates from historical data and are not guaranteed.

## How firm each number is

| Type | Statement |
|---|---|
| **Measured** | 96.8% / 97.0% match with the bot's labels on two held-out quarters. 77.2% bot vs closing-team agreement over 15 months. Rs 0 model cost per request. |
| **Estimated** | About Rs 1.15 lakh/month in current misroute cost, and about Rs 40,000/month that a final-team version might save. Both use policy §4 rates (Rs 305 per transfer, Rs 260 per extra contact) and historical transfer counts. |
| **Assumed** | The bot routed Jul–Sep 2026 the same way it did before. We expect 95.5–97.5% on those months, but cannot measure it until the labels exist. |
| **Needs the shadow run** | Live agreement with the bot; real transfer counts; whether low-confidence requests should go to triage. |

## What we suggest for next week

1. **Start a 2–4 week shadow run.** The new router runs alongside the bot on live requests without
   changing anything for agents. Switch the bot off if it matches the bot on at least 90% of live
   requests. Our expectation is about 96%.
2. **Get the monthly cost confirmed in writing for Farhan:** Rs 0/month for the model, plus
   whatever hosting IT chooses.
3. **Decide whether to keep "match the bot" as the goal.** The alternative is to aim for "reach the
   right team first time", which Meenal's team would feel directly. We can measure it during the
   same shadow run.
4. **For headcount planning, use the teams that close requests, not the bot's queue counts.** The
   bot's queues overstate Repairs, Billing and Filters & Consumables. They understate Installs &
   Demo, Returns, Warranty and Product Advice.

| Team | Closed per month (Jan–Jun 2026) | Bot queue per month |
|---|---|---|
| Repairs | ~165 | 206 |
| Returns & Replacement | ~106 | 81 |
| Installs & Demo | ~104 | 78 |
| Product Advice | ~90 | 77 |
| Billing | ~90 | 113 |
| Warranty Claims | ~89 | 67 |
| Filters & Consumables | ~73 | 95 |
