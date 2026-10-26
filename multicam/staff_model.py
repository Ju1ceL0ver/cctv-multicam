"""Strict train-only preprocessing and deployable staff classifier, raw masked ReID in."""
import numpy as np
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC


def norm(x):
    x=np.asarray(x,np.float32)
    return x/np.maximum(np.linalg.norm(x,axis=1,keepdims=True),1e-6)


def geometry(boxes,cam):
    b=np.asarray(boxes,np.float32)
    return np.stack([(b[:,0]+b[:,2])/2560,b[:,3]/720,(b[:,3]-b[:,1])/720,np.asarray(cam)=='cam2'],1)


class Features:
    def fit(self,c,s):
        self.cp=PCA(n_components=min(128,len(c)-1),svd_solver='randomized',random_state=0).fit(norm(c))
        self.sp=PCA(n_components=min(64,len(s)-1),svd_solver='randomized',random_state=0).fit(norm(s))
        return self
    def transform(self,c,s,g):
        return np.concatenate([norm(self.cp.transform(norm(c))),norm(self.sp.transform(norm(s))),g],1).astype(np.float32)


class RoleModel:
    def __init__(self,config):
        self.config=config
    def fit(self,x,y,raw):
        cfg=self.config;self.models=[]
        if cfg['kind']=='mlp':
            w=np.ones(len(y));w[y]=cfg.get('staff_weight',1)
            for seed in cfg.get('seeds',[0]):
                m=MLPClassifier(tuple(cfg.get('layers',[256,64])),alpha=cfg.get('alpha',.0001),
                                max_iter=400,early_stopping=True,random_state=seed)
                m.fit(x,y,sample_weight=w);self.models.append(m)
        elif cfg['kind']=='svm':
            m=SVC(C=cfg['C'],gamma=cfg['gamma'],class_weight={False:1,True:cfg.get('staff_weight',1)})
            m.fit(x,y);self.models=[m]
        elif cfg['kind']=='knn':
            m=KNeighborsClassifier(cfg['k'],weights='distance',metric='cosine',n_jobs=4)
            m.fit(raw,y);self.models=[m]
        return self
    def predict(self,x,raw):
        if self.config['kind']=='svm':
            v=self.models[0].decision_function(x)
            return 1/(1+np.exp(-np.clip(v,-50,50)))
        xx=raw if self.config['kind']=='knn' else x
        return np.mean([m.predict_proba(xx)[:,1] for m in self.models],axis=0)


class StaffClassifier:
    def __init__(self,features,models,weights,threshold=.5):
        self.features=features;self.models=models;self.weights=weights;self.threshold=threshold
    def predict_proba(self,cloth,shape,boxes,cam):
        x=self.features.transform(cloth,shape,geometry(boxes,cam))
        raw=np.concatenate([norm(cloth),norm(shape)],1)/np.sqrt(2)
        p=sum(w*m.predict(x,raw) for w,m in zip(self.weights,self.models))
        return np.stack([1-p,p],1)
    def predict(self,cloth,shape,boxes,cam):
        return self.predict_proba(cloth,shape,boxes,cam)[:,1]>=self.threshold


def load_current(root=None):
    """Load the newest supervised refit; old frozen benchmark models remain separate."""
    import pickle
    from pathlib import Path
    root=Path(root) if root is not None else Path(__file__).resolve().parent
    with (root/'data'/'staff'/'current_role.pkl').open('rb') as f:
        return pickle.load(f)
