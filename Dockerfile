FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt requirements-ml.txt ./
RUN pip install --no-cache-dir -r requirements-ml.txt
COPY backend ./backend
COPY ml ./ml
COPY data/transactions_synthetic.csv ./data/transactions_synthetic.csv
COPY reports ./reports
COPY api.py index.html script.js styles.css logo.png ./
# Mount local models/ and Hugging Face cache before enabling transformer inference.
COPY deployment ./deployment
ENV FINANCE_CLASSIFIER=transformer
CMD ["sh", "deployment/start.sh"]
