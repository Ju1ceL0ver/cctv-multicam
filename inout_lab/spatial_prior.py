"""Training-only scene priors: Gaussian coordinate channels + spatial kNN.
Never fit priors on the day being evaluated. Use day cross-fitting for training
features too: otherwise nearest-neighbor labels leak into their own inputs.
"""
import numpy as np
from scipy.spatial.distance import cdist

NAMES=('outside','inside','doorway')
SCALE=np.array([320/180,1.])

def anchors(A):
    A=np.atleast_2d(A)
    # upper 15% of visible mask, mask centroid, original tracker foot.
    return np.stack([A[:,234+16:234+18],A[:,10:12],A[:,6:8]],axis=1)

class SpatialPrior:
    def fit(self,A,y):
        self.points=anchors(A).astype(float); self.labels=np.asarray(y,dtype=int)
        self.means=np.empty((3,3,2));self.covariances=np.empty((3,3,2,2))
        for a in range(3):
            for c in range(3):
                p=self.points[self.labels==c,a]
                if len(p)<2:raise ValueError('Every class needs at least two training examples')
                self.means[a,c]=p.mean(0)
                self.covariances[a,c]=np.cov(p.T)+np.diag([(8/320)**2,(8/180)**2])
        self.inverse=np.linalg.inv(self.covariances)
        self.logdet=np.linalg.slogdet(self.covariances)[1]
        return self

    def transform(self,A,include_knn=True):
        p=anchors(A).astype(float);parts=[]
        for a in range(3):
            delta=p[:,a,None,:]-self.means[a]
            d2=np.einsum('nci,cij,ncj->nc',delta,self.inverse[a],delta)
            logdensity=-.5*(d2+self.logdet[a])
            probability=np.exp(logdensity-logdensity.max(1,keepdims=True));probability/=probability.sum(1,keepdims=True)
            parts.extend([delta.reshape(len(p),-1),d2,probability])
        if include_knn:
            for a in [0,1,2,None]:
                query=(p*SCALE).reshape(len(p),-1) if a is None else p[:,a]*SCALE
                reference=(self.points*SCALE).reshape(len(self.points),-1) if a is None else self.points[:,a]*SCALE
                distances=cdist(query,reference)
                count=min(61,len(reference));inds=np.argpartition(distances,count-1,axis=1)[:,:count]
                ds=np.take_along_axis(distances,inds,axis=1);order=np.argsort(ds,axis=1);inds=np.take_along_axis(inds,order,axis=1);ds=np.take_along_axis(ds,order,axis=1)
                labs=self.labels[inds]
                for k in [3,9,25,61]:
                    kk=min(k,count);weights=1/np.maximum(ds[:,:kk],.02)
                    votes=np.stack([(weights*(labs[:,:kk]==c)).sum(1) for c in range(3)],1)
                    votes=(votes+.05)/(votes.sum(1,keepdims=True)+.15)
                    entropy=-(votes*np.log(np.maximum(votes,1e-12))).sum(1,keepdims=True)
                    parts.extend([votes,ds[:,[0,kk-1]],entropy])
                parts.append(np.stack([np.min(distances[:,self.labels==c],axis=1) for c in range(3)],1))
        return np.column_stack(parts).astype(np.float32)

    def channels(self,shape=(180,320),anchor=0):
        """Three float32 probability-map channels, HxWx3, fixed camera scene.
        anchor 0=upper mask, 1=centroid, 2=tracker foot. Mirror/warp channels with
        image+mask when augmenting. Maps are priors, not a hard door boundary.
        """
        h,w=shape;yy,xx=np.indices(shape);p=np.stack([xx/w,yy/h],-1)
        delta=p[:,:,None,:]-self.means[anchor]
        d2=np.einsum('hwci,cij,hwcj->hwc',delta,self.inverse[anchor],delta)
        logdensity=-.5*(d2+self.logdet[anchor]);v=np.exp(logdensity-logdensity.max(2,keepdims=True));v/=v.sum(2,keepdims=True)
        return v.astype(np.float32)

    def summary(self):
        return {anchor:{name:{'mean_xy_1280x720':(self.means[a,c]*[1280,720]).tolist(),'n':int((self.labels==c).sum())} for c,name in enumerate(NAMES)} for a,anchor in enumerate(['upper_mask','mask_centroid','tracker_foot'])}

class SpatialFeatureClassifier:
    """Drop-in model for DoorClassifier; exposes maps and means for inspection."""
    def __init__(self,prior,classifier,limit=144):
        self.prior=prior;self.classifier=classifier;self.limit=limit
        self.classes_=classifier.classes_;self.n_jobs=1
    def predict_proba(self,A):
        self.classifier.n_jobs=self.n_jobs
        return self.classifier.predict_proba(np.column_stack([A,self.prior.transform(A,include_knn=self.limit>36)[:,:self.limit]]))
