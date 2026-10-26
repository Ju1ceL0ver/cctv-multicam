"""Frozen chronological staff experiment; choose on development, evaluate test once.
No test-day PCA, scaling, threshold selection or nearest labelled neighbours.
Run: python staff_audit.py SNAPSHOT OUTPUT
"""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(key, '4')
import json
import sys
import time
from pathlib import Path
import numpy as np
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from threadpoolctl import threadpool_limits


def norm(x):
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-6)


def metrics(y, p, src):
    q = p >= .5
    out = {'n': len(y), 'accuracy': float((q == y).mean()),
           'staff_as_customer': int((~q & y).sum()),
           'customer_as_staff': int((q & ~y).sum()),
           'staff_recall': float(q[y].mean()) if y.any() else None,
           'customer_recall': float((~q[~y]).mean()) if (~y).any() else None}
    out['by_source'] = {str(s): {'n': int((src == s).sum()), 'accuracy': float((q[src == s] == y[src == s]).mean())} for s in sorted(set(src))}
    return out


def run(snapshot, output):
    z = np.load(snapshot)
    c, s, y, days, src = norm(z['cloth']), norm(z['shape']), z['y'], z['days'], z['src']
    x = np.concatenate([c, s], axis=1) / np.sqrt(2)
    box = z['boxes']
    where = np.stack([(box[:, 0]+box[:, 2])/2560, box[:, 3]/720, (box[:, 3]-box[:, 1])/720, z['cam']=='cam2'], 1)
    unique = sorted(set(days))
    if len(unique) < 6:
        raise ValueError('Need at least six days')
    test_days, dev_days = unique[-2:], unique[-4:-2]
    train = ~np.isin(days, test_days + dev_days)
    dev, test = np.isin(days, dev_days), np.isin(days, test_days)
    report = {'snapshot': str(snapshot), 'n': len(y), 'train_days': unique[:-4], 'development_days': dev_days, 'test_days': test_days,
              'protocol': 'Chronological; last two days frozen before model comparison; selection by development accuracy only. Previously reviewed labels, not a new blind test. Repeated staff across days intentional for known-store deployment.', 'development': {}}
    factories = {
        'raw_logistic': lambda: LogisticRegression(C=1, max_iter=500, random_state=0),
        'raw_svm': lambda: SVC(C=10, kernel='rbf', gamma=1, probability=True, random_state=0),
        'raw_knn': lambda: KNeighborsClassifier(7, weights='distance', metric='cosine', n_jobs=4),
    }
    def prepared(tr, ev):
        cp = PCA(n_components=min(128, int(tr.sum())-1), svd_solver='randomized', random_state=0).fit(c[tr])
        sp = PCA(n_components=min(64, int(tr.sum())-1), svd_solver='randomized', random_state=0).fit(s[tr])
        def transform(ix):
            return np.concatenate([norm(cp.transform(c[ix])), norm(sp.transform(s[ix])), where[ix]], 1)
        return transform(tr), transform(ev)
    def fit_predict(name, tr, ev):
        if name == 'strict_current_mlp':
            a,b = prepared(tr,ev)
            model = MLPClassifier((256,64),max_iter=400,early_stopping=True,random_state=0)
        else:
            a,b=x[tr],x[ev]; model=factories[name]()
        t=time.time();model.fit(a,y[tr]);p=model.predict_proba(b)[:,1]
        return p, time.time()-t
    for name in ['strict_current_mlp']+list(factories):
        p,sec=fit_predict(name,train,dev)
        report['development'][name]=metrics(y[dev],p,src[dev]); report['development'][name]['seconds']=sec
        print(name,report['development'][name],flush=True)
        Path(output).write_text(json.dumps(report,indent=2),encoding='utf-8')
    winner=max(report['development'],key=lambda k:report['development'][k]['accuracy'])
    report['selected']=winner
    final_train=~test
    report['test']={}
    for name in dict.fromkeys(['strict_current_mlp',winner]):
        p,sec=fit_predict(name,final_train,test)
        report['test'][name]=metrics(y[test],p,src[test]); report['test'][name]['seconds']=sec
        report['test'][name]['by_day']={str(d):metrics(y[test][days[test]==d],p[days[test]==d],src[test][days[test]==d]) for d in test_days}
        np.savez_compressed(str(output)+'.'+name+'.npz', ids=z['ids'][test],y=y[test],p=p,days=days[test])
        print('TEST',name,report['test'][name],flush=True)
    report['complete']=True
    Path(output).write_text(json.dumps(report,indent=2),encoding='utf-8')

if __name__ == '__main__':
    with threadpool_limits(limits=4):
        run(sys.argv[1],sys.argv[2])
