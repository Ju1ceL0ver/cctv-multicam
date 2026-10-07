"""Переразметка двери в два класса. Запуск: python relabel2.py  -> http://localhost:8765
Показывает 190 'в проёме' + 47 'не понять'. 1 внутри, 2 снаружи, 0 пропустить, Z назад, ←/→ листать.
Ответы -> relabel2.json (индекс строки в io_cam1.npz -> 'inside'|'outside'). Исходные метки не меняются."""
import json, sys, io
from pathlib import Path
from http.server import BaseHTTPRequestHandler, HTTPServer
import numpy as np, cv2
R = Path(__file__).parent
z = np.load(R/'io_cam1.npz'); F = z['F']; X = z['X']; ids = z['ids']; day = z['day']
orig = np.load(R/'diagnostics/original_labels.npy')
todo = [int(i) for i in np.flatnonzero((orig == 3) | (orig == 0))]
todo.sort(key=lambda i: (day[i], ids[i]))
OUT = R/'relabel2.json'
labels = json.loads(OUT.read_text()) if OUT.exists() else {}

def png(a):
    return cv2.imencode('.png', a)[1].tobytes()
def frame(i):
    f = F[i][:, :, :3][:, :, ::-1].copy()
    m = (F[i][:, :, 3] > 127).astype('uint8')
    cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    f = cv2.resize(f, (960, 540), interpolation=cv2.INTER_CUBIC)
    for c in cs: cv2.polylines(f, [(c*3).astype('int32')], True, (0, 255, 0), 2)
    return png(f)
def crop(i):
    c = X[i][:, :, :3][:, :, ::-1]
    return png(cv2.resize(c, (288, 480), interpolation=cv2.INTER_CUBIC))

PAGE = """<!doctype html><meta charset=utf-8><title>Два класса</title>
<style>body{background:#111;color:#eee;font:16px sans-serif;margin:12px}img{vertical-align:top}
.b{display:inline-block;padding:2px 8px;border-radius:4px}#st{font-size:20px;margin:6px 0}</style>
<div id=st></div><img id=f><img id=c><div>1 — внутри · 2 — снаружи · 0 — не могу решить · Z — назад · ←/→ листать</div>
<script>
let N=%d, items=%s, lab=%s, k=0;
function show(){ if(k>=items.length)k=items.length-1; if(k<0)k=0; const i=items[k];
 f.src='/f/'+i; c.src='/c/'+i;
 const done=Object.keys(lab).length;
 st.innerHTML=(k+1)+' / '+items.length+' · размечено '+done+' · сейчас: <b>'+(lab[i]||'—')+'</b>'; }
async function put(v){ const i=items[k]; if(v) lab[i]=v; else delete lab[i];
 await fetch('/set',{method:'POST',body:JSON.stringify({i:i,v:v})}); if(v){k++;} show(); }
addEventListener('keydown',e=>{ if(e.key=='1')put('inside'); else if(e.key=='2')put('outside');
 else if(e.key=='0'){k++;show();} else if(e.key=='z'||e.key=='я'){k--;put(null);} 
 else if(e.key=='ArrowRight'){k++;show();} else if(e.key=='ArrowLeft'){k--;show();}});
k=Math.max(0,items.findIndex(i=>!lab[i])); show();
</script>"""

class H(BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def send(self, b, t):
        self.send_response(200); self.send_header('Content-Type', t); self.end_headers(); self.wfile.write(b)
    def do_GET(self):
        p = self.path
        if p == '/': self.send((PAGE % (len(todo), json.dumps(todo), json.dumps(labels))).encode(), 'text/html; charset=utf-8')
        elif p.startswith('/f/'): self.send(frame(int(p[3:])), 'image/png')
        elif p.startswith('/c/'): self.send(crop(int(p[3:])), 'image/png')
        else: self.send_response(404); self.end_headers()
    def do_POST(self):
        d = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
        if d['v']: labels[str(d['i'])] = d['v']
        else: labels.pop(str(d['i']), None)
        OUT.write_text(json.dumps(labels, indent=0)); self.send(b'ok', 'text/plain')

if __name__ == '__main__':
    print(len(todo), 'примеров; открой http://localhost:8765'); HTTPServer(('127.0.0.1', 8765), H).serve_forever()
