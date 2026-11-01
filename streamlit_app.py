"""Streamlit client for the local inference/analytics API; no credentials in the UI."""
import json
import os
from pathlib import Path
import urllib.error
import urllib.request
import pandas as pd
import plotly.express as px
import streamlit as st
from backend.analytics import CATEGORIES

ROOT = Path(__file__).resolve().parent
API_URL = os.environ.get('FINANCE_API_URL','http://127.0.0.1:8001').rstrip('/')


def api(path,payload=None):
    request = urllib.request.Request(API_URL+path,
        data=json.dumps(payload).encode() if payload is not None else None,
        headers={'Content-Type':'application/json'})
    with urllib.request.urlopen(request,timeout=90) as response:
        return json.load(response)


st.set_page_config(page_title='Finance Analyzer | NLP & Spending Intelligence',page_icon='📊',layout='wide')
st.title('Finance Analyzer')
st.caption('Transaction classification, spending patterns, and traceable financial insights')
st.info('Portfolio demo: transformer and embedding models are evaluated on public synthetic bank descriptions. Anomaly flags require human review.')
try:
    health = api('/api/health')
    st.sidebar.success(f"Inference API connected · {health['classifier']}")
except (OSError,ValueError,urllib.error.HTTPError):
    st.error('Start the inference service first: FINANCE_CLASSIFIER=transformer PORT=8001 python api.py')
    st.stop()
if 'transactions' not in st.session_state:
    st.session_state.transactions = []
if 'analysis' not in st.session_state:
    st.session_state.analysis = None

uploaded = st.sidebar.file_uploader('Upload transactions CSV',type=['csv'])
if uploaded is not None:
    content = uploaded.getvalue().decode('utf-8-sig')
    if st.sidebar.button('Validate and import'):
        try:
            result = api('/api/import',{'csv':content})
            existing = {t['id']:t for t in st.session_state.transactions}
            existing.update({t['id']:t for t in result['transactions']})
            st.session_state.transactions = list(existing.values())
            st.session_state.analysis = None
            st.sidebar.success(f"{result['accepted']} new; {result['duplicates']} duplicates; {len(result['rejected'])} rejected")
            if result['rejected']:
                st.sidebar.dataframe(result['rejected'])
        except (OSError,ValueError,urllib.error.HTTPError):
            st.error('Import failed. Check the required CSV columns: Date, Description, Amount, optional Category.')
if st.sidebar.button('Load sample transactions'):
    # Deterministic synthetic data for charts/anomaly exploration; not model-training data.
    from ml.demo import demo_rows
    st.session_state.transactions = demo_rows()
    st.session_state.analysis = None
    st.rerun()
st.sidebar.caption('Uploaded records remain in this browser session; validated imports also persist in the local API database. Public deployment needs isolated per-user storage.')

rows = st.session_state.transactions
if not rows:
    st.write('Upload a CSV or load sample transactions to begin. Positive amounts indicate income; negative amounts indicate expenses.')
    st.stop()

classify_tab, dashboard_tab, evaluation_tab = st.tabs(['Classify & review','Spending intelligence','Evaluation'])
with classify_tab:
    if st.button('Categorize unreviewed transactions'):
        pending = [r for r in rows if r['category']=='Other']
        if pending:
            try:
                with st.spinner('Running inference through the API…'):
                    predictions = api('/api/classify',{'transactions':pending})['predictions']
                for row,prediction in zip(pending,predictions):
                    row['prediction'] = prediction
                    if not prediction['needs_review']:
                        row['category'] = prediction['category']
                st.session_state.analysis = None
                st.success(f"{len(predictions)} predictions; {sum(p['needs_review'] for p in predictions)} require review")
            except (OSError,ValueError,urllib.error.HTTPError):
                st.error('Classification failed. Confirm that trained model files are available to the API.')
        else:
            st.info('No uncategorized transactions remain.')
    display = pd.DataFrame([{**{k:r[k] for k in ('id','date','description','amount','category')},
        'suggested_category':r.get('prediction',{}).get('category',''),
        'confidence':r.get('prediction',{}).get('confidence'),
        'needs_review':r.get('prediction',{}).get('needs_review',False)} for r in rows])
    st.dataframe(display,hide_index=True,width='stretch')
    st.caption('Confidence is uncalibrated; the 0.65 review threshold is illustrative.')
    with st.form('review'):
        selected_id = st.selectbox('Transaction to correct',[r['id'] for r in rows],
                                   format_func=lambda identity:next(r['description'] for r in rows if r['id']==identity))
        category = st.selectbox('Reviewed category',CATEGORIES)
        if st.form_submit_button('Save correction'):
            transaction = next(r for r in rows if r['id']==selected_id)
            try:
                api('/api/corrections',{'transaction':transaction,'category':category})
                transaction['category'] = category
                transaction.pop('prediction',None)
                st.session_state.analysis = None
                st.rerun()
            except (OSError,ValueError,urllib.error.HTTPError):
                st.error('Correction was not saved; please retry once the API is available.')
with dashboard_tab:
    try:
        if st.session_state.analysis is None:
            st.session_state.analysis = api('/api/analyze',{'transactions':rows})
        report = st.session_state.analysis
        a,b,c = st.columns(3)
        a.metric('Recorded income',f"₹{report['metrics']['income']:,.2f}")
        b.metric('Recorded expenses',f"₹{report['metrics']['expenses']:,.2f}")
        c.metric('Net cash flow',f"₹{report['metrics']['net_cash_flow']:,.2f}")
        st.caption(f"Period: {report['period']['start']} to {report['period']['end']} · {report['count']} unique transactions")
        monthly = pd.DataFrame(report['monthly'])
        monthly['expenses'] = monthly['expense_paise']/100
        st.plotly_chart(px.line(monthly,x='month',y='expenses',markers=True,title='Monthly recorded expenses'),width='stretch')
        category_data = pd.DataFrame(report['categories'])
        if not category_data.empty:
            category_data['expenses'] = category_data['expense_paise']/100
            st.plotly_chart(px.bar(category_data,x='category',y='expenses',title='Category spending'),width='stretch')
        st.subheader('Evidence-backed insights')
        for insight in report['insights']:
            st.write(insight['claim'])
            if insight.get('transaction_ids'):
                with st.expander('Supporting transaction IDs'):
                    st.write(insight['transaction_ids'])
        st.subheader('Unusual spending to review')
        anomalies = report['anomalies']
        if anomalies['available']:
            st.write(f"{len(anomalies['flags'])} review flags across {len([r for r in rows if r['amount']<0])} expenses")
            st.dataframe(anomalies['flags'],hide_index=True,width='stretch')
            st.dataframe(anomalies['patterns'],hide_index=True,width='stretch')
            st.caption(' · '.join(anomalies['limitations']))
        else:
            st.info(anomalies['reason'])
        st.subheader('Budget forecast')
        if report['forecast']['available']:
            st.metric(report['forecast']['month'],f"₹{report['forecast']['forecast_inr']:,.2f}")
            st.json(report['forecast'])
        else:
            st.info(report['forecast']['reason'])
        st.download_button('Download evidence report',json.dumps(report,indent=2),'financial-evidence.json','application/json')
    except (OSError,ValueError,urllib.error.HTTPError):
        st.error('Analytics API is unavailable; your current session data is retained.')
with evaluation_tab:
    path = ROOT/'reports/transfer_learning_evaluation.json'
    if path.exists():
        benchmark = json.loads(path.read_text())
        st.caption(benchmark['dataset']['provenance'])
        st.dataframe([{'model':name,'accuracy':result['accuracy'],'macro_f1':result['macro_f1']}
                       for name,result in benchmark['results'].items()],hide_index=True)
        st.write(f"Training examples: {benchmark['training_labels']}; held-out test examples: {benchmark['test_examples']}")
        st.json(benchmark['limitations'])
        st.download_button('Download benchmark',path.read_text(),'transfer-learning-evaluation.json')
    else:
        st.info('Run python -m ml.prepare and python -m ml.train to generate measured results.')
