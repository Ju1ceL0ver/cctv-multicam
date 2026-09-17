"""One-keystroke labeller for identity-break candidates.

Each candidate is "track A vanished, track B appeared nearby moments later".
The page shows A's LAST crops next to B's FIRST crops -- the frames either side
of the break, which is exactly the comparison a tracker has to make -- and asks
one question: same person?

    1 = same person (a real identity break the tracker should have bridged)
    2 = different people (the tracker was right to open a new id)
    3 = can't tell
    Backspace = back one

Answers go to data/tracklets/break_labels.json and are saved on every key, so
closing the tab loses nothing. This file is the first ground truth for ReID
on this camera: any descriptor -- OSNet, a colour histogram, a distilled
student -- is scored against the same pairs.
"""
import os, json, glob
from flask import Flask, jsonify, request, send_file, abort

HOME = r'C:\Users\ArykovAA'
ROOT = os.path.join(HOME, 'cctv_ai', 'retail_analytics')
TR = os.path.join(ROOT, 'data', 'tracklets')
LABELS = os.path.join(TR, 'break_labels.json')
SHOW = 8


def load_pairs():
    pairs = []
    for m in sorted(glob.glob(os.path.join(TR, '*', '*', 'meta.json'))):
        d = json.load(open(m, encoding='utf-8'))
        if not d.get('completed'):
            continue
        vdir = os.path.dirname(m)
        rel = os.path.relpath(vdir, TR)
        for b in d['breaks']:
            lost = sorted(glob.glob(os.path.join(vdir, 't%06d' % b['lost_track'], '*.jpg')))
            new = sorted(glob.glob(os.path.join(vdir, 't%06d' % b['new_track'], '*.jpg')))
            if not lost or not new:
                continue
            key = '%s|%d|%d' % (rel.replace(os.sep, '/'), b['lost_track'], b['new_track'])
            pairs.append({'key': key, 'gap_s': b['gap_seconds'], 'dist': b['distance_ref_px'],
                          'lost': [os.path.relpath(p, TR) for p in lost[-SHOW:]],
                          'new': [os.path.relpath(p, TR) for p in new[:SHOW]]})
    return pairs


PAIRS = load_pairs()
app = Flask(__name__)


def labels():
    try:
        return json.load(open(LABELS, encoding='utf-8'))
    except Exception:
        return {}


@app.get('/api/pairs')
def api_pairs():
    return jsonify({'pairs': PAIRS, 'labels': labels()})


@app.post('/api/label')
def api_label():
    body = request.get_json(force=True)
    data = labels()
    if body.get('answer') is None:
        data.pop(body['key'], None)
    else:
        data[body['key']] = body['answer']
    tmp = LABELS + '.tmp'
    json.dump(data, open(tmp, 'w', encoding='utf-8'), indent=1)
    os.replace(tmp, LABELS)
    return jsonify({'ok': True, 'done': len(data)})


@app.get('/img/<path:rel>')
def img(rel):
    path = os.path.normpath(os.path.join(TR, rel))
    if not path.startswith(TR) or not os.path.exists(path):
        abort(404)
    return send_file(path)


PAGE = r'''<!doctype html><meta charset="utf-8"><title>Same person?</title>
<style>
body{font:15px system-ui;margin:0;background:#15171a;color:#e8e8e8}
header{padding:10px 16px;background:#222;display:flex;gap:18px;align-items:center}
.row{display:flex;gap:6px;align-items:flex-end;padding:8px 16px;flex-wrap:wrap}
.row img{height:220px;border-radius:4px;image-rendering:auto;background:#000}
h3{margin:10px 16px 0;font-weight:500;color:#aaa}
.keys b{background:#333;padding:2px 7px;border-radius:4px;margin-right:4px}
#ans{font-weight:600}
</style>
<header><span id="pos"></span><span id="meta"></span><span id="ans"></span>
<span class="keys"><b>1</b>same <b>2</b>different <b>3</b>can't tell <b>⌫</b>back <b>→</b>skip</span></header>
<h3>A — last crops before the track vanished</h3><div class="row" id="lost"></div>
<h3>B — first crops of the new track</h3><div class="row" id="new"></div>
<script>
let P=[],L={},i=0;
const names={same:'SAME',different:'DIFFERENT',unsure:"CAN'T TELL"};
function show(){
  if(i>=P.length){document.getElementById('pos').textContent='all '+P.length+' done';return}
  const p=P[i];
  document.getElementById('pos').textContent=(i+1)+' / '+P.length+'  (labelled '+Object.keys(L).length+')';
  document.getElementById('meta').textContent='gap '+p.gap_s+' s, dist '+p.dist+' px';
  document.getElementById('ans').textContent=L[p.key]?('→ '+names[L[p.key]]):'';
  for(const s of ['lost','new']){const el=document.getElementById(s);el.innerHTML='';
    for(const r of p[s]){const im=new Image();im.src='/img/'+r;el.appendChild(im)}}
}
async function put(a){const p=P[i];
  await fetch('/api/label',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({key:p.key,answer:a})});
  if(a)L[p.key]=a;else delete L[p.key];}
document.onkeydown=async e=>{
  const m={'1':'same','2':'different','3':'unsure'}[e.key];
  if(m&&i<P.length){await put(m);i++;show()}
  else if(e.key==='Backspace'){i=Math.max(0,i-1);show();e.preventDefault()}
  else if(e.key==='ArrowRight'){i=Math.min(P.length,i+1);show()}
};
fetch('/api/pairs').then(r=>r.json()).then(d=>{P=d.pairs;L=d.labels;
  i=P.findIndex(p=>!L[p.key]);if(i<0)i=P.length;show()});
</script>'''


@app.get('/')
def index():
    return PAGE


if __name__ == '__main__':
    print('pairs with crops on both sides:', len(PAIRS))
    app.run(host='127.0.0.1', port=5060, threaded=True)
