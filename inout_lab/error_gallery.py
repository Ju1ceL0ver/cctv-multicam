"""Local error review; does not change dataset labels."""
import base64,io,json,argparse
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
parser=argparse.ArgumentParser();parser.add_argument('--split',choices=['development18','validation19'],default='development18');args=parser.parse_args();split=args.split
root=Path(__file__).parent;z=np.load(root/'io_cam1.npz');p=np.load(root/'diagnostics/predictions.npz')
frames=z['F'];labels=z['y'];ids=z['ids']
names=['снаружи','внутри','в проёме'];models=[k[:-len('_'+split)] for k in p.files if k.endswith('_'+split)]
ix=p[split];best=max(models,key=lambda m:(p[m+'_development18'].argmax(1)==labels[p['development18']]).mean())
probs=p[best+'_'+split];bad=np.where(probs.argmax(1)!=labels[ix])[0];bad=sorted(bad,key=lambda j:-probs[j].max())
rows=[]
for j in bad:
 i=ix[j];f=frames[i];img=Image.fromarray(f[:,:,:3]);a=np.asarray(img).copy();mask=f[:,:,3]>64;a[mask]=(a[mask]*.5+np.array([0,255,80])*.5).astype('uint8');img=Image.fromarray(a).resize((640,360));buf=io.BytesIO();img.save(buf,format='JPEG')
 rows.append(f'<article><p>{ids[i]} · разметка: <b>{names[labels[i]]}</b> · {best}: <b>{names[probs[j].argmax()]}</b> ({probs[j].max():.0%})</p><img src="data:image/jpeg;base64,{base64.b64encode(buf.getvalue()).decode()}"></article>')
html='<!doctype html><meta charset="utf-8"><title>Ошибки 18.09</title><style>body{font:16px sans-serif;background:#16181c;color:#eee;margin:32px}article{display:inline-block;margin:12px;max-width:640px}img{width:100%}p{line-height:1.5}</style>'+f'<h1>Ошибки 18.09: {len(bad)} / {len(ix)}</h1><p>Лучший вариант по 18.09: {best}. Этот день использован для подбора. Метки не изменены.</p>'+''.join(rows)
(root/('diagnostics/errors18.html' if split=='development18' else 'diagnostics/errors19.html')).write_text(html.replace('Ошибки 18.09','Ошибки 19.09').replace('Этот день использован для подбора.', '19.09 исключён из обучения; вариант выбран по 18.09.') if split=='validation19' else html)
print(best,len(bad))
