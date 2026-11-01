"""Validated transaction ingestion and SQL-backed evidence. Amounts use integer paise."""
import csv
import hashlib
import io
import json
import sqlite3
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation

CATEGORIES = ['Transport', 'Food & Drink', 'Shopping', 'Entertainment', 'Groceries',
              'Utilities', 'Income', 'Travel', 'Electronics', 'Fuel', 'Health',
              'Fitness', 'Rent', 'Education', 'Fees', 'Other', 'Restaurants', 'Transportation',
              'Healthcare', 'Subscription', 'Insurance', 'Mortgage', 'Personal Care', 'Transfer']


def connect(path):
    db = sqlite3.connect(path)
    db.row_factory = sqlite3.Row
    db.executescript('''
    CREATE TABLE IF NOT EXISTS transactions (
      id TEXT PRIMARY KEY, date TEXT NOT NULL, description TEXT NOT NULL,
      amount_paise INTEGER NOT NULL, category TEXT NOT NULL);
    CREATE TABLE IF NOT EXISTS predictions (
      id INTEGER PRIMARY KEY, transaction_id TEXT NOT NULL, category TEXT NOT NULL,
      confidence REAL NOT NULL, model_version TEXT NOT NULL, created_at TEXT DEFAULT CURRENT_TIMESTAMP);
    CREATE TABLE IF NOT EXISTS corrections (
      id INTEGER PRIMARY KEY, transaction_id TEXT NOT NULL, old_category TEXT,
      new_category TEXT NOT NULL, created_at TEXT DEFAULT CURRENT_TIMESTAMP);
    ''')
    return db


def normalize(row):
    raw_date = row.get('date')
    if isinstance(raw_date, (int, float)):
        date = datetime.fromtimestamp(raw_date / 1000, timezone.utc).date().isoformat()
    else:
        date = None
        for fmt in ('%Y-%m-%d', '%d/%m/%Y', '%d-%m-%Y'):
            try:
                date = datetime.strptime(str(raw_date), fmt).date().isoformat()
                break
            except ValueError:
                pass
        if date is None:
            raise ValueError('Date must be YYYY-MM-DD or DD/MM/YYYY or DD-MM-YYYY')
    description = str(row.get('description') or '').strip()
    if not description or len(description) > 1000:
        raise ValueError('Description must contain 1-1000 characters')
    try:
        amount = Decimal(str(row.get('amount', '')).replace(',', '').replace('₹', '').replace('$', ''))
        if not amount.is_finite() or amount != amount.quantize(Decimal('.01')) or abs(amount) > 10**12:
            raise ValueError('Amount must be finite, at most 2 decimal places, and within supported range')
    except InvalidOperation as exc:
        raise ValueError('Invalid amount') from exc
    category = row.get('category') or 'Other'
    if category not in CATEGORIES:
        raise ValueError('Unknown category')
    paise = int(amount * 100)
    identity = json.dumps([date, description.casefold(), paise], ensure_ascii=False)
    return dict(id=hashlib.sha256(identity.encode()).hexdigest()[:24], date=date,
                description=description, amount_paise=paise, category=category)


def ingest(db, rows):
    accepted, rejected, duplicates = [], [], 0
    with db:
        for number, row in enumerate(rows, 1):
            try:
                t = normalize(row)
                cursor = db.execute('INSERT OR IGNORE INTO transactions VALUES (:id,:date,:description,:amount_paise,:category)', t)
                if cursor.rowcount:
                    accepted.append(t)
                else:
                    duplicates += 1
            except (ValueError, TypeError, OverflowError) as exc:
                rejected.append({'row': number, 'reason': str(exc)})
    return {'accepted': len(accepted), 'duplicates': duplicates, 'rejected': rejected}


def parse_csv(text):
    reader = csv.DictReader(io.StringIO(text.lstrip('\ufeff')))
    fields = {str(k).strip().lower(): k for k in reader.fieldnames or []}
    if not {'date', 'description', 'amount'} <= fields.keys():
        raise ValueError('CSV requires Date, Description, Amount; Category is optional')
    return [{k: row.get(original) for k, original in fields.items()} for row in reader]


def transactions(db):
    return [dict(id=r['id'], date=r['date'], description=r['description'],
                 amount=r['amount_paise']/100, category=r['category'])
            for r in db.execute('SELECT * FROM transactions ORDER BY date,id')]


def snapshot(db):
    totals = dict(db.execute('''SELECT COUNT(*) count, MIN(date) start, MAX(date) end,
      COALESCE(SUM(CASE WHEN amount_paise>0 THEN amount_paise ELSE 0 END),0) income_paise,
      COALESCE(SUM(CASE WHEN amount_paise<0 THEN -amount_paise ELSE 0 END),0) expense_paise
      FROM transactions''').fetchone())
    categories = [dict(r) for r in db.execute('''SELECT category, SUM(-amount_paise) expense_paise,
      COUNT(*) count FROM transactions WHERE amount_paise<0 GROUP BY category ORDER BY expense_paise DESC,category''')]
    monthly = [dict(r) for r in db.execute('''WITH months AS (
      SELECT substr(date,1,7) month, SUM(CASE WHEN amount_paise<0 THEN -amount_paise ELSE 0 END) expense_paise
      FROM transactions GROUP BY 1)
      SELECT *, expense_paise-LAG(expense_paise) OVER (ORDER BY month) change_paise FROM months ORDER BY month''')]
    evidence = [dict(r) for r in db.execute('SELECT id,date,amount_paise,category FROM transactions ORDER BY ABS(amount_paise) DESC,id LIMIT 5')]
    income, expense = totals['income_paise'], totals['expense_paise']
    metrics = {'income': income/100, 'expenses': expense/100, 'net_cash_flow': (income-expense)/100,
               'savings_rate_percent': round((income-expense)/income*100, 2) if income else None}
    insights = []
    if totals['count']:
        insights.append({'claim': f"Recorded net cash flow is INR {metrics['net_cash_flow']:.2f}.",
                         'metric': 'net_cash_flow', 'value': metrics['net_cash_flow']})
        if categories:
            top = categories[0]
            ids = [r[0] for r in db.execute('SELECT id FROM transactions WHERE category=? AND amount_paise<0 ORDER BY id', (top['category'],))]
            insights.append({'claim': f"Largest expense category: {top['category']} (INR {top['expense_paise']/100:.2f}).",
                             'metric': 'largest_expense_category', 'value': top['expense_paise']/100,
                             'transaction_ids': ids})
    return {'period': {'start': totals['start'], 'end': totals['end']}, 'count': totals['count'],
            'metrics': metrics, 'categories': categories, 'monthly': monthly, 'evidence': evidence,
            'insights': insights, 'limitations': [
                'Covers recorded transactions only; missing accounts or dates can change conclusions.',
                'Monthly totals may cover partial months. No credit-risk or investment suitability inference is made.']}
