"""Read-only deployment checks. Never send personal data to a deployment probe."""
import argparse
import json
from urllib.request import Request,urlopen


def verify(base):
    base = base.rstrip('/')
    with urlopen(base+'/api/health',timeout=40) as response:
        health = json.load(response)
    with urlopen(base+'/api/evaluation',timeout=40) as response:
        evaluation = json.load(response)
    assert health['status']=='ok'
    assert health['classifier']=='transformer',health
    assert evaluation['results']['fine_tuned_transformer']['encoder_weights_changed']
    print(json.dumps({'url':base,'health':health,'evaluated_test_examples':evaluation['test_examples']},indent=2))


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('url')
    verify(parser.parse_args().url)
