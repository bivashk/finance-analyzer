import {test} from 'node:test';
import assert from 'node:assert/strict';
import gateway from '../netlify/functions/finance-api.mjs';

test('gateway rejects missing configuration and unsupported routes',async()=>{
  delete process.env.FINANCE_API_URL;
  assert.equal((await gateway(new Request('https://frontend.test/api/health'))).status,503);
  process.env.FINANCE_API_URL='https://backend.test';
  assert.equal((await gateway(new Request('https://frontend.test/api/.env'))).status,404);
});

test('gateway keeps configured secret on server and forwards browser session',async()=>{
  process.env.FINANCE_API_URL='https://backend.test';
  process.env.FINANCE_PROXY_TOKEN='server-secret';
  const original=globalThis.fetch;
  globalThis.fetch=async(url,options)=>{
    assert.equal(String(url),'https://backend.test/api/classify');
    assert.equal(options.headers['X-Finance-Session'],'anonymous-session');
    assert.equal(options.headers['X-Finance-Proxy'],'server-secret');
    return Response.json({predictions:[]});
  };
  try{
    const response=await gateway(new Request('https://frontend.test/api/classify',{
      method:'POST',headers:{'X-Finance-Session':'anonymous-session','Content-Type':'application/json'},body:'{}'}));
    assert.equal(response.status,200);
    assert.equal(response.headers.get('X-Finance-Proxy'),null);
  }finally{globalThis.fetch=original;delete process.env.FINANCE_PROXY_TOKEN;delete process.env.FINANCE_API_URL;}
});
