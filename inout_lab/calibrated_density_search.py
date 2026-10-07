"""Camera-scale / head-size reference learned from visible training feet.
Physical scale is in calibration units. No metric tile size is assumed.
All three classes and whole-day exclusion; pose is used only for calibration.
"""
from pathlib import Path
import json
import cv2
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier,ExtraTreesClassifier
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).parent

class CameraScale:
    def __init__(self,cal,head,geometry,rich,pose256,pose512):
        self.K=np.asarray(cal['K'],float);self.dist=np.asarray(cal['dist'],float);self.rv=np.asarray(cal['rvec'],float);self.tv=np.asarray(cal['tvec'],float)
        self.R=cv2.Rodrigues(self.rv)[0];self.C=-self.R.T@self.tv;self.H=float(self.C[2]);self.head=head;self.g=geometry;self.rich=rich
        footray=self.rays(geometry[:,6:8]);lam=-self.H/np.where(np.abs(footray[:,2])>.001,footray[:,2],-.001)
        self.floor=self.C+lam[:,None]*footray
        visible=np.zeros(len(head),bool)
        for pose in [pose256,pose512]:
            pp=pose[:,:51].reshape(-1,17,3);ankles=pp[:,15:17,:2].mean(1);err=np.linalg.norm((ankles-geometry[:,6:8])*[1280,720],axis=1)
            visible|=(pp[:,15:17,2].min(1)>.5)&(err<20)
        self.visible=visible&(lam>0)
        self.references=[]
        for j in range(4):
            h=head[:,8*j:8*j+8];center=h[:,:2];ray=self.rays(center)
            left=np.column_stack([h[:,4],h[:,1]]);right=np.column_stack([h[:,6],h[:,1]])
            width=np.linalg.norm(self.undistort(right)-self.undistort(left),axis=1)
            delta=self.floor[:,:2]-self.C[:2];head_lam=(delta*ray[:,:2]).sum(1)/np.maximum((ray[:,:2]**2).sum(1),1e-5)
            head_height=(self.C[2]+head_lam*ray[:,2])/self.H
            ratio=width*head_lam/self.H
            good=self.visible&(width>.001)&(head_lam>0)&(head_height>.1)&(head_height<.7)&(ratio>.005)&(ratio<.3)
            self.references.append((center,ray,width,ratio,good))
    def undistort(self,points):
        return cv2.undistortPoints((points*[2560,1440]).reshape(-1,1,2).astype(float),self.K,self.dist).reshape(-1,2)
    def rays(self,points):
        n=self.undistort(points);return np.column_stack([n,np.ones(len(n))])@self.R
    def transform(self,tr):
        parts=[];audit=[]
        for center,ray,width,ratio,good in self.references:
            values=ratio[tr][good[tr]]
            if len(values)<10:raise RuntimeError('Too few physically usable head/foot references')
            ratios=np.quantile(values,[.25,.5,.75]);audit.append({'count':len(values),'ratios':ratios.tolist()})
            for ref in ratios:
                lam=self.H*ref/np.maximum(width,.001);xyz=self.C+lam[:,None]*ray;head_z=xyz[:,2]/self.H;xyz[:,2]=0
                foot=cv2.projectPoints(xyz,self.rv,self.tv,self.K,self.dist)[0].reshape(-1,2)/[2560,1440]
                density=(self.K[0,0]/2/(lam/self.H))**2
                physical_area=self.rich[:,9]*1280*720/np.maximum(density,1)
                physical_height=self.g[:,5]*720/np.sqrt(np.maximum(density,1))
                parts.append(np.column_stack([np.log1p(density),physical_area,physical_height,np.clip(head_z,-5,5),np.clip(foot,-5,5)]))
        features=np.column_stack(parts).astype(np.float32);assert np.isfinite(features).all()
        return features,audit

if __name__=='__main__':
    z=np.load(ROOT/'io_cam1.npz');y=z['y'];days=z['day'];valid=np.load(ROOT/'diagnostics/original_labels.npy')!=0
    A=np.load(ROOT/'diagnostics/aligned_features.npy')[:,:1584];G=np.load(ROOT/'diagnostics/aligned_geometry.npy');cal=json.loads((ROOT/'diagnostics/calib_final_source.json').read_text())['cam1']
    scale=CameraScale(cal,A[:,234:936],G,A[:,:234],np.load(ROOT/'diagnostics/pose_features.npy'),np.load(ROOT/'diagnostics/pose_512_features.npy'))
    P={k:np.zeros((len(y),3)) for k in ['geometry','context']};audits={}
    with threadpool_limits(limits=2):
        for d in np.unique(days):
            tr=np.where(valid&(days!=d))[0];te=days==d;S,audit=scale.transform(tr);audits[str(d)]=audit
            for name in P:
                B=np.column_stack([G,S,A[:,:39]]) if name=='geometry' else np.column_stack([A,S])
                model=HistGradientBoostingClassifier(max_iter=160,max_leaf_nodes=15,min_samples_leaf=10,l2_regularization=3,random_state=19) if name=='geometry' else ExtraTreesClassifier(n_estimators=160,max_features=.7,random_state=19,n_jobs=2)
                P[name][te]=model.fit(B[tr],y[tr]).predict_proba(B[te])
            print('camera density evaluated',d,'references',[a['count'] for a in audit],flush=True)
    report={'calibration':audits,'models':{}}
    for name,prob in P.items():
        p=prob.argmax(1);gross=valid&(y<2)&(p<2)&(p!=y)
        report['models'][name]={'defined':int(valid.sum()),'accuracy':float((p[valid]==y[valid]).mean()),'wrong':int((valid&(p!=y)).sum()),'gross':int(gross.sum()),'gross_indices':np.where(gross)[0].tolist(),'doorway_recall':float((p[y==2]==2).mean())}
        np.savez(ROOT/f'diagnostics/calibrated_density_{name}_predictions.npz',probabilities=prob)
    (ROOT/'calibrated_density_report.json').write_text(json.dumps(report,indent=2));print('CALIBRATED DENSITY RESULT',json.dumps(report['models']),flush=True)
