# Finance Analyzer: reviewable transaction intelligence

## Business problem and decision

A banking analytics workflow needs reliable categories and explainable spending summaries without treating every model suggestion as correct. The decision supported here is which suggestions can be automatically applied and which need human review. Customer behavior profiles support budgeting conversations; they do not determine credit eligibility, fraud, or retention interventions.

## Delivery

The existing HTML/JavaScript and Plotly frontend remains the product interface. Python provides validated CSV ingestion, SQL cash-flow and category/month aggregations, transformer inference, calibration, corrections, customer behavior features, and session-scoped operational metrics. A Netlify server-side gateway connects the static frontend to a Render model API without exposing the gateway credential in JavaScript.

Public-demo mode processes uploaded transaction rows in memory and discards them after each request. Anonymous session-scoped telemetry stores prediction categories, confidence, review outcomes, transaction hashes, and inference duration with a 24-hour reporting window; expired metadata is removed on the next metrics access. Descriptions, amounts, and uploaded customer histories are excluded from telemetry. Browser transaction persistence remains controlled by the user's existing settings. This is a portfolio demonstration, not a production banking platform.

## Dataset and evaluation design

The public source contains 68,000 MIT-licensed synthetic US bank-format descriptions across 17 categories. It is not real customer data. Source: https://huggingface.co/datasets/DoDataThings/us-bank-transaction-categories-v2.

The earlier normalized-description-group benchmark achieved 88.35% transformer accuracy, but allowed merchant overlap. A new experiment retrains from pretrained BERT-tiny, not from that previously fine-tuned checkpoint. Public generator vocabulary and conservative alias matching identify merchant families; ambiguous and unmatched descriptions are excluded. There are 528 training families, 190 validation families, and 181 test families with no matched-family overlap. Each class contributes 100 training, 50 validation, and 50 test examples: 1,700/850/850 total.

This is a stronger matched-family test, not proof of perfect merchant identification. The vocabulary and matching assumptions are recorded, unknown aliases may be missed, and synthetic templates can still recur across partitions.

## Measured classification results

| Model | Merchant-holdout accuracy | Macro-F1 |
| --- | ---: | ---: |
| TF-IDF + logistic regression | 58.59% | 0.559 |
| Frozen MiniLM embeddings + logistic regression | 63.06% | 0.627 |
| Fine-tuned BERT-tiny | 60.71% | 0.602 |

The stronger test exposes merchant memorization and ambiguity. BERT does not outperform frozen embeddings on this holdout. The earlier 88.35% is retained with its original split qualification, not substituted for this result.

## Review policy and workload tradeoff

The validation set is divided by merchant family: 365 rows fit a scalar temperature and 485 rows select the review threshold. A 0.67 confidence threshold met an empirical 95% accepted-prediction accuracy target with sufficient validation observations.

On the held-out test set, the calibrated transformer automatically accepts **39.29%** of predictions, with **95.81% accuracy within that accepted subset**, and sends **60.71%** for manual review. Overall classifier accuracy remains **60.71%**. The accepted-subset figure must never be presented as overall 95.81% accuracy or as a production guarantee.

Epoch selection and post-hoc calibration share validation data, so a completely separate calibration cohort would strengthen the design. Error analysis includes per-category reports, confusion matrices, confidence reliability bins, and synthetic misclassification examples. Common confusions include Groceries vs Shopping and Personal Care vs Entertainment/Subscription.

## Customer intelligence

Monthly SQL-backed features cover transaction frequency, average expenses, recorded surplus, expense variability, discretionary share, and recurring-expense share. Rule-based profiles provide understandable baselines. Standardized KMeans segmentation chooses the number of groups using silhouette separation and reports stability across seeds plus agreement with the rule baseline.

A deliberately structured synthetic cohort of 36 customers produces four groups, silhouette 0.928 and mean seed-stability ARI 1.0. These demonstrate implementation and repeatability, not business validity or retention lift. A real deployment would require representative cohorts, feature governance, stability over time, and stakeholder review.

## Operational evidence

The dashboard measures event-level auto-accept coverage, manual-review rate, corrections among reviewed predictions, median/95th-percentile batch-amortized inference time, and optional configured compute-cost estimates. Empty metrics remain unavailable instead of displaying invented business outcomes. Review agreement is biased by which predictions users choose to review and is not reported as unbiased model accuracy.

No user study, measured time savings, revenue increase, retention uplift, or 40% insight-discovery improvement has been established. The paired-study protocol and evaluator are available separately for collecting real consented observations.

## Fit to the target role

The project demonstrates business framing, Python and SQL, exploratory feature construction, supervised NLP, transformer transfer learning, sentence embeddings, validation and calibration, customer segmentation, human review, deployment integration, and communication of measured limitations. The next evidence gap is representative real-world validation and an observed business outcome, not another library added to the stack.

## Reproduce and review

- `python -m ml.prepare` verifies the source corpus and original split.
- `python -m ml.merchant_prepare` creates the conservative merchant-family holdout using the downloaded generator vocabulary.
- `python -m ml.train --dataset data/public/merchant_prepared.csv --output models/merchant --report-path reports/merchant_evaluation.json --metadata reports/merchant_dataset.json --epochs 8` retrains all three approaches.
- `python -m ml.calibrate` fits temperature and the review policy using validation data only, then reports held-out results.
- `python -m ml.customer_demo` creates the synthetic segmentation report.
- `reports/merchant_dataset.json`, `merchant_evaluation.json`, `calibration_evaluation.json`, and `customer_segmentation.json` contain the underlying evidence.

## Verified delivery

The deployed frontend is https://financeanalyzerbivash.netlify.app and its model API is https://finance-analyzer-api-mdqo.onrender.com on Render’s free plan. The release source is https://github.com/bivashk/finance-analyzer/tree/codex/jd-aligned-analytics. Synthetic production probes verified calibrated model inference, financial totals, correction telemetry, session isolation, and all 36 synthetic customer profiles. Deployment IDs and timestamps are recorded in `reports/deployment_evidence.json`. Free-service restarts can reset ephemeral metrics; 24 hours is a maximum retention period, not a persistence guarantee.
