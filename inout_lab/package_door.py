from pathlib import Path
import numpy as np,json,pickle,time,base64,io
from PIL import Image
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from door_ensemble import DoorEnsemble
from door_features import expanded_features
from door_classifier import DoorClassifier
root=Path(__file__).parent;z=np.load(root/'io_cam1.npz');F=z['F'];X=z['X'];G=z['G'];y=z['y'];days=z['day'];door=z['door'];valid=np.load(root/'diagnostics/original_labels.npy')!=0;A=np.column_stack([np.load(root/('diagnostics/'+n+'_features.npy')) for n in ['rich','head','context']])
diff=max(float(np.abs(expanded_features(F[i],X[i,:,:,3],G[i,1:])-A[i]).max()) for i in range(0,len(y),17));assert diff<1e-6,diff
model=ExtraTreesClassifier(n_estimators=160,min_samples_leaf=1,max_features=.7,random_state=19,n_jobs=1).fit(A[valid],y[valid]);boost=HistGradientBoostingClassifier(max_iter=160,max_leaf_nodes=15,min_samples_leaf=10,l2_regularization=3,random_state=19).fit(A[valid],y[valid]);model=DoorEnsemble(model,boost);path=root/'door_classifier.pkl';path.write_bytes(pickle.dumps(model,protocol=5));m=DoorClassifier();m.predict_prepared(F[0],X[0,:,:,3],G[0,1:]);start=time.perf_counter()
for i in range(100):m.predict_prepared(F[i],X[i,:,:,3],G[i,1:])
ms=(time.perf_counter()-start)*10;p=(np.load(root/'diagnostics/context_predictions.npz')['probabilities']+np.load(root/'diagnostics/context_boost.npz')['probabilities'])/2;pred=p.argmax(1);gross=valid&(y<2)&(pred<2)&(pred!=y);r={'defined':int(valid.sum()),'doorway':int((y==2).sum()),'accuracy':float((pred[valid]==y[valid]).mean()),'gross_errors':int(gross.sum()),'bytes':path.stat().st_size,'prepared_ms':ms,'feature_parity':diff,'thresholds':{}}
for t in [.0,.6,.7,.8,.85,.9,.95]:
 ok=valid&(p.max(1)>=t);r['thresholds'][str(t)]={'coverage':float(ok.sum()/valid.sum()),'door_coverage':float((ok&door).sum()/(valid&door).sum()),'accuracy_accepted':float((pred[ok]==y[ok]).mean()),'gross_errors':int((gross&ok).sum()),'wrong':int(((pred!=y)&ok).sum())}
(root/'door_classifier_report.json').write_text(json.dumps(r,indent=2));names=['снаружи','внутри','в проёме'];rows=[];bad=np.where(valid&(pred!=y))[0];bad=sorted(bad,key=lambda i:(not gross[i],-p[i].max()))
for i in bad:
 a=F[i,:,:,:3].copy();mask=F[i,:,:,3]>32;a[mask]=(a[mask]*.45+np.array([0,255,90])*.55).astype('uint8');buf=io.BytesIO();Image.fromarray(a).resize((640,360)).save(buf,format='JPEG');rows.append(f'<article><p>{z["ids"][i]}<br>Разметка: <b>{names[y[i]]}</b> · модель: <b>{names[pred[i]]}</b> ({p[i].max():.1%}) · {"ГРУБАЯ ОШИБКА" if gross[i] else "ошибка класса"}</p><img src="data:image/jpeg;base64,{base64.b64encode(buf.getvalue()).decode()}"></article>')
(root/'door_errors.html').write_text('<!doctype html><meta charset="utf-8"><title>Новый классификатор: ошибки</title><style>body{background:#17191f;color:#eee;font:16px sans-serif;margin:24px}article{display:inline-block;width:640px;margin:12px;vertical-align:top}img{width:100%}</style><h1>4 грубых ошибки; все 95 ошибок трёх классов</h1><p>Каждый день исключён из обучения модели, которая его предсказывала. Все примеры «в проёме» сохранены. Подбор вариантов проводился по этим дням: это не слепой тест. Метки не изменены.</p>'+''.join(rows));print(json.dumps(r,indent=2),flush=True)
