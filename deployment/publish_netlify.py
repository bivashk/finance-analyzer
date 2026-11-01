"""Atomic Netlify draft deployment via the documented digest/functions API.
Credentials are read locally and never printed or included in public assets.
"""
import argparse
import hashlib
import io
import json
import os
import time
import zipfile
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request,urlopen

ROOT = Path(__file__).resolve().parents[1]
BASE = 'https://api.netlify.com/api/v1'


def credentials():
    if os.environ.get('NETLIFY_AUTH_TOKEN'):
        return os.environ['NETLIFY_AUTH_TOKEN']
    config = json.loads((Path.home()/'Library/Preferences/netlify/config.json').read_text())
    return config['users'][config['userId']]['auth']['token']


def call(method,path,body=None,binary=False):
    if body is not None and not binary:
        body = json.dumps(body).encode()
    request = Request(BASE+path,data=body,method=method,headers={
        'Authorization':'Bearer '+credentials(),
        'Content-Type':'application/octet-stream' if binary else 'application/json'})
    with urlopen(request,timeout=45) as response:
        raw = response.read()
    return json.loads(raw) if raw else None


def package_gateway():
    wrapper = '''exports.handler = async (event) => {
  const gateway = (await import('./gateway.mjs')).default;
  const headers = new Headers(event.headers || {});
  const body = event.body ? (event.isBase64Encoded ? Buffer.from(event.body,'base64').toString() : event.body) : undefined;
  const request = new Request('https://' + (headers.get('host') || 'localhost') + event.path, {
    method:event.httpMethod, headers,
    body:['GET','HEAD'].includes(event.httpMethod) ? undefined : body
  });
  const result = await gateway(request);
  return {statusCode:result.status,headers:Object.fromEntries(result.headers),body:await result.text()};
};
'''
    archive = io.BytesIO()
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as zipped:
        zipped.writestr('finance-api.js',wrapper)
        zipped.writestr('gateway.mjs',(ROOT/'netlify/functions/finance-api.mjs').read_bytes())
    return archive.getvalue()


def deploy(site_id,api_url):
    site = call('GET',f'/sites/{site_id}')
    if site.get('name')!='financeanalyzerbivash':
        raise ValueError('Unexpected target site; refusing deployment')
    secret = dict(line.split('=',1) for line in (ROOT/'.env.deployment').read_text().splitlines() if '=' in line)['FINANCE_PROXY_TOKEN']
    files = { '/'+p.name:p.read_bytes() for p in (ROOT/'public').iterdir() if p.is_file() }
    gateway = package_gateway()
    digest = {name:hashlib.sha1(content).hexdigest() for name,content in files.items()}
    function_sha = hashlib.sha256(gateway).hexdigest()
    environment = [{'key':key,'value':value,'is_secret':True,'scopes':['functions']}
                   for key,value in [('FINANCE_API_URL',api_url),('FINANCE_PROXY_TOKEN',secret)]]
    info = call('POST',f'/sites/{site_id}/deploys?title=Calibrated%20NLP%20and%20customer%20intelligence',
                {'files':digest,'functions':{'finance-api':function_sha},'draft':True,'environment':environment})
    deploy_id = info['id']
    state_path = ROOT/'reports/netlify_deployment.json'
    def persist(current):
        safe = {k:current.get(k) for k in ('id','state','deploy_url','deploy_ssl_url','url','ssl_url','error_message')}
        safe['site_id'] = site_id
        state_path.write_text(json.dumps(safe,indent=2)+'\n')
        return safe
    persist(info)
    print(json.dumps({'draft_deploy_id':deploy_id,'state':info['state']}),flush=True)
    for name,sha in digest.items():
        if sha in info.get('required',[]):
            call('PUT',f'/deploys/{deploy_id}/files{name}',files[name],True)
    if function_sha in info.get('required_functions',[]):
        call('PUT',f'/deploys/{deploy_id}/functions/finance-api?runtime=js',gateway,True)
    for _ in range(60):
        info = call('GET',f'/deploys/{deploy_id}')
        persist(info)
        if info['state'] in ('ready','error'):
            print(json.dumps(persist(info)),flush=True)
            if info['state']=='error':
                raise RuntimeError('Netlify deploy failed')
            return
        time.sleep(2)
    raise RuntimeError('Draft deployment is still processing; inspect saved deployment ID')


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--site-id',required=True)
    parser.add_argument('--api-url',required=True)
    args = parser.parse_args()
    deploy(args.site_id,args.api_url)
