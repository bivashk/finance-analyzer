// Same-origin gateway: secrets remain in server-side environment variables.
const ALLOWED = new Set(['health','evaluation','metrics','customer-demo','import','analyze','chat','classify','corrections','segments','cohort-import']);
export default async (request) => {
  const backend = process.env.FINANCE_API_URL;
  if (!backend) return Response.json({error:'The model API is not configured yet.'},{status:503});
  const incoming = new URL(request.url);
  const route = incoming.pathname.replace(/^\/api\//,'').replace(/^\/\.netlify\/functions\/finance-api\//,'');
  if (!ALLOWED.has(route) || !['GET','POST'].includes(request.method)) return new Response('Not found',{status:404});
  let base;
  try { base = new URL(backend); } catch { return new Response('Invalid API configuration',{status:503}); }
  if (base.protocol!=='https:') return new Response('The API must use HTTPS',{status:503});
  const headers = {'Content-Type':'application/json','X-Finance-Session':request.headers.get('X-Finance-Session') || ''};
  if (process.env.FINANCE_PROXY_TOKEN) headers['X-Finance-Proxy']=process.env.FINANCE_PROXY_TOKEN;
  try {
    const body = request.method==='POST' ? await request.text() : undefined;
    if (body && new TextEncoder().encode(body).length>5_000_000) return new Response('Request too large',{status:413});
    const result = await fetch(new URL(`/api/${route}`,base),{method:request.method,headers,body,signal:AbortSignal.timeout(25000)});
    return new Response(result.body,{status:result.status,headers:{'Content-Type':'application/json','Cache-Control':'no-store'}});
  } catch { return Response.json({error:'The model API is waking up or unavailable. Please retry.'},{status:503}); }
};
export const config={path:'/api/*'};
