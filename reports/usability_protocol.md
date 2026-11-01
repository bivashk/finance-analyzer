# Insight-discovery study — not yet conducted

No 40% improvement has been measured. This protocol defines how to collect evidence, rather than assigning a desired result to the application.

Recruit at least 10 consenting participants. Use anonymous IDs and synthetic transactions. Each participant uses both interfaces: (A) a basic transaction table and (B) the Streamlit charts, calculated insights, and anomaly review. Randomize which condition comes first, and use matched but distinct datasets to limit memorization. Record the actual randomized assignment separately; do not expose participant identities or personal financial data in the repository.

Define and freeze ten tasks and answer keys before recruitment: identify largest spending category, compare monthly expenses, find largest payment, calculate cash flow, identify recurring expenses, and investigate designated unusual spending cases. An answer counts as a correct insight only if it matches the answer key and cites relevant transactions. Have a reviewer score answers without seeing condition labels. Keep the same task count, time limit, and scoring rules across conditions.

Record one summary row for each participant and condition:

```csv
participant_id,condition,tasks_total,correct_insights,seconds
```

Conditions must be `baseline` and `enhanced`. Do not fill the file with simulated people. After collecting observations, run:

```sh
python -m ml.evaluate_study /path/to/consented-anonymous-results.csv
```

The evaluator requires complete matched pairs and at least 10 participants. It reports mean correct-insight rates, absolute percentage-point difference with a paired bootstrap interval, relative rate improvement, and mean task-time reduction. These are distinct measures: faster completion is not automatically improved discovery. A small convenience sample cannot establish a population-wide causal effect. Report negative and inconclusive outcomes too.
