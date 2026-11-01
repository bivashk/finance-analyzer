# Finance Analyzer — NLP & Spending Intelligence

Python, Hugging Face Transformers, Sentence Transformers, scikit-learn, SQLite, HTML/JavaScript, and Plotly. The project fine-tunes a pretrained BERT encoder, compares it with frozen sentence embeddings and TF-IDF, serves classification through an HTTP API, and provides spending patterns, anomaly review, forecasts, and traceable insights.

## Earlier text-group benchmark (merchant overlap allowed)

| Model | Test accuracy | Macro-F1 |
| --- | ---: | ---: |
| Fine-tuned BERT-tiny | 88.35% | 87.98% |
| TF-IDF + logistic regression | 84.71% | 84.41% |
| Frozen MiniLM sentence embeddings + logistic regression | 80.35% | 80.03% |

These results use **850 held-out synthetic examples**, with **1,700 training examples (100 per category)** and 510 validation examples across 17 categories. All encoder parameters are fine-tuned for BERT-tiny; its saved embedding-weight checksums confirm that pretrained weights changed. MiniLM is frozen: only its logistic-regression classification head is trained.

Source: [DoDataThings/us-bank-transaction-categories-v2](https://huggingface.co/datasets/DoDataThings/us-bank-transaction-categories-v2), a MIT-licensed corpus of 68,000 **synthetic** bank-format descriptions. This is not validation on real customer transactions. BERT-tiny is a small pretrained language encoder, not a generative LLM. No 95% accuracy or 40% user-impact claim is supported.

Details, model revisions, source checksums, split counts, per-category results, confusion matrices, training losses, validation scores, and encoder-weight hashes are recorded in `reports/public_dataset.json` and `reports/transfer_learning_evaluation.json`. The environment used for this run is recorded in `reports/environment-lock.txt`.

## Prepare and train

Requires Python 3.11+ and a modern browser. Initial setup downloads public datasets and pretrained model weights. Training runs on CPU with four PyTorch threads; no user financial data is uploaded.

```sh
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-ml.txt
export HF_HOME="$PWD/.cache/huggingface"
export HF_HUB_DISABLE_XET=1
python -m ml.prepare
python -m ml.train --epochs 8
```

The preparation step pins the corpus revision, verifies downloaded checksums, discards conflicting normalized-text groups, and creates seed-42 train/validation/test partitions. Each normalized-description group belongs to only one split. **Merchant identities and templates can still overlap**: this is not a merchant-disjoint real-world test. Training is capped at 100 examples per category; this is a defined label budget, not proof that these are the minimum sufficient labels. Model selection uses validation macro-F1, with a single final test evaluation. Do not tune on the reported test results to reach a desired resume number.

`models/transformer/` contains the fine-tuned safetensors checkpoint, tokenizer, and provenance. `models/sentence/head.json` contains a portable JSON logistic head and the pinned MiniLM model revision. Model/data downloads are ignored by Git; reproduce or explicitly package them before moving to another machine. Remote custom code is disabled. Sentence embeddings are generated with the actual `sentence-transformers/all-MiniLM-L6-v2` model.

## Run the original frontend

The original HTML/JavaScript dashboard used by the Netlify site is the primary interface.

```sh
source .venv/bin/activate
export HF_HOME="$PWD/.cache/huggingface"
HF_HUB_OFFLINE=1 FINANCE_MODEL_PATH="$PWD/models/merchant/transformer" FINANCE_CLASSIFIER=transformer PORT=8001 python api.py
```

Open http://127.0.0.1:8001. The original dashboard and trained-model API run together. Upload a CSV with `Date,Description,Amount` and optional `Category`. Supported dates: YYYY-MM-DD, DD/MM/YYYY, DD-MM-YYYY. Positive amounts are income; negative amounts are expenses. For classification, omit Category or use Other.

The dashboard calls `/api/import`, `/api/classify`, `/api/corrections`, and `/api/analyze`. Low-confidence predictions remain uncategorized for human review. Corrections are logged; they do not automatically retrain the model. The 0.65 review threshold is illustrative and probabilities are uncalibrated.

Switch to `FINANCE_CLASSIFIER=sentence` for MiniLM or `FINANCE_CLASSIFIER=tfidf` for the lightweight model. The lightweight workflow only needs `requirements.txt`; run `python -m backend.model`, then `python api.py` and open http://127.0.0.1:8000.

The Streamlit experiment is retained as an optional alternative in `streamlit_app.py`, not the recommended interface. Its separate dependencies are in `requirements-streamlit.txt`.

## Spending intelligence

- Validated ingestion handles CSV quoting, missing/invalid values, duplicate detection, row-level rejections, and exact integer-paise storage.
- SQL provides category/month aggregations and period-over-period changes using a window function. Insights include the period and supporting transaction IDs.
- Plotly charts show monthly and category spending. Statistical patterns include category totals, counts, and median expense size.
- Robust log-amount modified z-scores and Isolation Forest flag unusual expense amount/weekday combinations. These are **unsupervised review suggestions, not confirmed fraud or errors**. Detection needs at least 20 expenses and is fitted on the submitted batch.
- Expense forecasting requires at least eight consecutive past months, excludes the current month, and compares last-month expenses with linear trend using chronological rolling-origin backtests. Its range uses the largest recent backtest error; it is not a calibrated prediction interval.
- Optional GenAI chat can select only verified statements. Calculations are independent of the LLM; invalid selections fall back to calculated evidence. Set `GROQ_API_KEY` and optional `FINANCE_LLM_MODEL` on the server to enable it. Questions and aggregate statements are sent to the provider only when enabled. API keys never enter browser storage.

## Validation

```sh
python -m unittest discover -s tests -v
node --test tests/frontend.test.cjs
```

Tests cover money/date validation, quoted CSVs, duplicates, unsupported GenAI claims, chronological forecasts, normalized split leakage, anomaly review, study-input validation, API contracts, private-file access protection, and Streamlit/Plotly integration. Streamlit tests are skipped when its optional dependency is absent. HTTP integration tests use loopback and temporary databases.

## Deployment and privacy

Live frontend: https://financeanalyzerbivash.netlify.app. Model service: https://finance-analyzer-api-mdqo.onrender.com. The calibrated model API is deployed on Render and connected through Netlify’s server-side gateway. Netlify copies only public frontend assets. The public `/api/*` gateway invokes the calibrated Python model on Render; the local service can still serve the frontend and API together.

A local container configuration is included:

```sh
# First produce the models and cache using the training commands above.
docker compose up --build
```

This serves the original frontend and API together, mounts model weights/cache read-only, stores imports in a named volume, and exposes only `127.0.0.1:8001`. Docker is not installed in the implementation environment, so the container configuration has not been runtime-tested.

The default local mode uses a **single-user local database**. The public-demo mode described below instead processes financial rows in request-local memory and isolates anonymous telemetry by session. It is a portfolio demo; production banking use still requires authentication, governance, and a representative real-world evaluation. The server serves only an explicit frontend asset allowlist, rejects cross-origin writes, and does not expose `.env`, database, backend source, or configuration files.

Duplicate identity is date + case-insensitive description + amount; two legitimate identical same-day payments can collide. Production ingestion should use bank-issued transaction IDs. Local SQLite defaults to `data/finance.sqlite3` (`FINANCE_DB` overrides it). Analysis/chat use exactly the active browser list. Browser-row deletion does not delete persisted audit records; stop the service and remove the local SQLite file to reset all imported history.

Previously exposed provider credentials must still be rotated in provider consoles. Removing credential text cannot invalidate a key or erase Git history. No banking eligibility, credit-risk, investment suitability, or confirmed fraud conclusions are produced.

## Resume evidence and remaining claims

See `reports/resume_bullets.tex` for wording supported by the measured run. Use current development dates for the new work.

An insight-discovery improvement requires real consented observations. `reports/usability_protocol.md` defines a paired, counterbalanced study with frozen tasks and answer keys. `python -m ml.evaluate_study results.csv` reports actual correct-insight rates, uncertainty, and task-time reduction after at least 10 complete participant pairs. No study observations have been collected and no uplift is claimed.

## JD-aligned customer intelligence and operational evidence

The original frontend now includes **Customer intelligence** and **Model performance** tabs. The first supports individual behavior profiles, anonymous Customer_ID CSV cohort imports, a 36-customer synthetic demo, rule baselines, KMeans selection, silhouette separation, and seed-stability evidence. The second reports session-scoped coverage, review/correction rates, batch-amortized inference latency, and optional configured compute-cost estimates. Financial histories are not used to infer protected traits, eligibility, creditworthiness, or confirmed fraud.

The stronger validation run uses a conservative matched-merchant-family holdout with no identified-family overlap. Overall transformer accuracy is **60.71%**; sentence embeddings achieve **63.06%**. Validation-only temperature calibration and a 0.67 review threshold yield **95.81% accepted-subset test accuracy at 39.29% coverage**. This is not overall 95.81% accuracy. See `reports/business_case_study.md` for the business framing, comparison, tradeoff, and limitations.

To serve the calibrated model with the original frontend locally:

```sh
export HF_HOME="$PWD/.cache/huggingface"
FINANCE_MODEL_PATH="$PWD/models/merchant/transformer" FINANCE_CLASSIFIER=transformer PORT=8003 python api.py
```

For the public-demo storage policy, add `FINANCE_PUBLIC_DEMO=1`. Uploaded financial rows then use request-local in-memory databases. Anonymous prediction metadata is isolated by a random browser-session header with a 24-hour reporting window; expired metadata is deleted on the next metrics access. `FINANCE_METRICS_DB` controls the telemetry database. Configure `FINANCE_COST_PER_COMPUTE_SECOND` only with a known rate; reported costs are compute-time estimates, never cloud-billing measurements.

Netlify's server-side gateway needs `FINANCE_API_URL` (HTTPS Render API URL) and `FINANCE_PROXY_TOKEN`. The same secret is set on the API. Never place it in a frontend config file. `deployment/Dockerfile.render` uses CPU PyTorch and the actual calibrated safetensors checkpoint. `render.yaml` describes a free-plan service. The calibrated model artifacts are included in the release branch and Render image; the main workspace models directory remains ignored by Git. Only the matched-merchant transformer needs to be packaged. The public service is a rate-limited stateless portfolio demo, not authenticated production banking infrastructure.

```sh
python3 deployment/build_frontend.py
node --test tests/frontend.test.cjs tests/proxy.test.mjs
python -m unittest discover -s tests -v
python3 deployment/verify_public.py https://your-netlify-site.netlify.app
```

Deployment source: https://github.com/bivashk/finance-analyzer/tree/codex/jd-aligned-analytics. `deployment/publish_netlify.py` creates a draft with deploy-scoped server-only credentials; verify it, then promote atomically. Future deployments made through other tools must configure equivalent gateway environment variables. `deployment/smoke_test.py` verifies actual calibrated inference, financial totals, correction telemetry, session isolation, and synthetic customer segmentation through the public gateway.
