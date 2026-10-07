from pathlib import Path
import numpy as np,base64,io
from PIL import Image
root=Path(__file__).parent;out=root/'diagnostics';z=np.load(root/'io_cam1.npz');f=z['F'];labels=z['y'];ids=z['ids'];days=z['day'];door=z['door'];p=np.load(out/'lodo_predictions.npz');prob=p['probabilities'];valid=p['valid'];pred=prob.argmax(1);names=['снаружи','внутри','в проёме'];bad=np.where(valid&(pred!=labels))[0];bad=sorted(bad,key=lambda i:(not door[i],-prob[i].max()));rows=[]
for i in bad:
 a=f[i,:,:,:3].copy();mask=f[i,:,:,3]>32;a[mask]=(a[mask]*.4+np.array([0,255,90])*.6).astype('uint8');buf=io.BytesIO();Image.fromarray(a).resize((640,360)).save(buf,format='JPEG')
 rows.append(f'<article><p>{ids[i]}<br>Разметка: <b>{names[labels[i]]}</b> · модель: <b>{names[pred[i]]}</b> ({prob[i].max():.0%})</p><img src="data:image/jpeg;base64,{base64.b64encode(buf.getvalue()).decode()}"></article>')
(out/'lodo_errors.html').write_text('<!doctype html><meta charset="utf-8"><title>Ошибки проверки по дням</title><style>body{font:16px sans-serif;margin:24px;background:#17191f;color:#eee}article{display:inline-block;vertical-align:top;width:640px;margin:12px}img{width:100%}p{line-height:1.6}</style><h1>Ошибки проверки по дням</h1><p>Каждый день предсказан моделью, обученной на других днях. Ответы «не понять» исключены заранее. Метки не изменены. Первыми идут ошибки у двери.</p>'+''.join(rows))
print(len(bad),'errors saved')
