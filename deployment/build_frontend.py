"""Copy only public assets; wire /api to a server-side Netlify function."""
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
public = ROOT/'public'
public.mkdir(exist_ok=True)
for name in ('index.html','script.js','styles.css','logo.png'):
    shutil.copyfile(ROOT/name,public/name)
(public/'_redirects').write_text('/api/* /.netlify/functions/finance-api/:splat 200\n')
