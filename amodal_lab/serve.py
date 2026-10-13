"""Local visibility annotation; binds only to localhost."""
from pathlib import Path
import json, os, time, threading, base64, io, hashlib
from PIL import Image
import numpy as np
from flask import Flask,request,jsonify,send_from_directory
ROOT=Path(__file__).resolve().parent
DATA=ROOT/'data'; LABELS=DATA/'labels.json';lock=threading.Lock()
app=Flask(__name__)
app.config["MAX_CONTENT_LENGTH"]=12*1024*1024
@app.get('/')
def home():return send_from_directory(ROOT,'index.html')
@app.get('/api/items')
def items():
 m=json.loads((DATA/'manifest.json').read_text());m['labels']=json.loads(LABELS.read_text()) if LABELS.exists() else {};return jsonify(m)
@app.get('/data/<path:name>')
def asset(name):return send_from_directory(DATA,name)
@app.post('/api/label')
def label():
 if request.headers.get('Origin') and request.headers['Origin']!=request.host_url.rstrip('/') :return jsonify(error='origin'),403
 p=request.get_json(silent=True)
 if not isinstance(p,dict):return jsonify(error='invalid JSON'),400
 ident=p.get('id');v=p.get('visibility');quality=p.get('quality')
 valid={i['id']:i for i in json.loads((DATA/'manifest.json').read_text())['items']}
 if ident not in valid or v not in ['full','occluded','edge','both','unclear'] or quality not in ['good','bad','unclear']:return jsonify(error='invalid label'),400
 correction=None
 if p.get('mask_png'):
  try:
   prefix,encoded=p['mask_png'].split(',',1)
   if prefix!='data:image/png;base64':raise ValueError('PNG required')
   raw=base64.b64decode(encoded,validate=True)
   with Image.open(io.BytesIO(raw)) as img:
    if img.format!='PNG':raise ValueError('PNG required')
    expected=Image.open(DATA/valid[ident]['mask']).size
    if img.size!=expected:raise ValueError('Размер маски не совпадает с исходным')
    a=np.asarray(img.convert('L'))
    if not np.isin(a,[0,255]).all():raise ValueError('Маска должна быть бинарной')
    if quality=='good' and not a.any():raise ValueError('Пустая маска не может быть хорошей')
    buf=io.BytesIO();Image.fromarray(a).save(buf,format='PNG');correction=buf.getvalue()
  except (ValueError,TypeError,OSError,Image.DecompressionBombError) as e:return jsonify(error=str(e)),400
 with lock:
  labels=json.loads(LABELS.read_text()) if LABELS.exists() else {}
  previous=labels.get(ident,{})
  record=dict(visibility=v,quality=quality,at=time.time())
  if correction is not None:
   digest=hashlib.sha256(correction).hexdigest()
   folder=DATA/'edited';folder.mkdir(exist_ok=True)
   name=f'edited/{ident}_{digest}.png';dest=DATA/name
   if not dest.exists():
    tmpmask=dest.with_suffix('.tmp');tmpmask.write_bytes(correction);os.replace(tmpmask,dest)
   record.update(mask_path=name,mask_sha256=digest,mask_edited=True)
  elif not p.get('reset_mask'):
   for key in ['mask_path','mask_sha256','mask_edited']:
    if key in previous:record[key]=previous[key]
  target=DATA/record.get('mask_path',valid[ident]['mask'])
  if quality=='good' and not np.asarray(Image.open(target).convert('L')).any():return jsonify(error='Пустая маска не может быть хорошей'),400
  labels[ident]=record
  tmp=LABELS.with_suffix('.tmp');tmp.write_text(json.dumps(labels,ensure_ascii=False,indent=2));os.replace(tmp,LABELS)
  with (DATA/'history.jsonl').open('a') as f:f.write(json.dumps(dict(id=ident,**labels[ident]))+'\n')
 return jsonify(ok=True,label=record)
if __name__=='__main__':app.run(host='127.0.0.1',port=8767,threaded=True)
