"""Synthetic-only public integration probe; never sends user financial records."""
import argparse
import json
import uuid
from urllib.request import Request,urlopen


def verify(base):
    session = uuid.uuid4().hex
    def call(path,payload=None,session_id=session):
        request = Request(base.rstrip('/')+path,
            data=json.dumps(payload).encode() if payload is not None else None,
            headers={'Content-Type':'application/json','X-Finance-Session':session_id})
        with urlopen(request,timeout=40) as response:
            return json.load(response)
    imported = call('/api/import',{'csv':'Date,Description,Amount\n2025-01-01,SALARY PAYROLL,50000\n2025-01-02,STARBUCKS,-250\n'})
    assert imported['accepted']==2
    rows = imported['transactions']
    predictions = call('/api/classify',{'transactions':rows})['predictions']
    assert len(predictions)==2 and all(p.get('calibrated') for p in predictions)
    assert all(p['model_version'].startswith('bert-tiny-finetuned-') for p in predictions)
    corrected = {**rows[0],'prediction':predictions[0]}
    call('/api/corrections',{'transaction':corrected,'category':predictions[0]['category']})
    metrics = call('/api/metrics')
    assert metrics['prediction_events']==2 and metrics['reviewed_predictions']==1
    other = call('/api/metrics',session_id=uuid.uuid4().hex)
    assert other['prediction_events']==0
    analysis = call('/api/analyze',{'transactions':rows})
    assert analysis['metrics']['net_cash_flow']==49750
    assert analysis['customer_profile']['income_recorded']
    demo = call('/api/customer-demo')
    segments = call('/api/segments',{'customers':demo['customers']})
    assert len(segments['customers'])==36
    return {'url':base,'calibrated_inference':True,'prediction_events':metrics['prediction_events'],
            'session_isolation':True,'financial_totals':True,'synthetic_customer_count':36,
            'segmentation_clusters':segments['chosen_k']}


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('url')
    report = verify(parser.parse_args().url)
    print(json.dumps(report,indent=2))
