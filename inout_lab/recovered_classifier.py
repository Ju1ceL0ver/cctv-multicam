"""Experimental camera-1 three-class MLP. No abstention, no temporal/pose/depth input.
Model evaluation belongs to owner_boundary_mlp_recovered_report.json (LODO),
not to the all-data refit used by this inference API.
"""
from pathlib import Path
import json,numpy as np,cv2,torch
from inside_outside import InsideOutside
from head_scale_mlp_search import ScaleExperts
from head_scale_search import mask_head_scale_features
from mask_density_features import density_features
from owner_boundary_search import boundary_features
from door_features import expanded_features
from crop_feature_search import crop_features
from align_visible_bbox import crop_bounds
R=Path(__file__).parent

def feature_parts(A,g,original):
    A=np.atleast_2d(A).astype('float32');g=np.atleast_2d(g).astype('float32');original=np.atleast_2d(original).astype('float32');r=A[:,:234];h=A[:,234:936];c=A[:,936:1584];crop=A[:,1584:]
    cal=json.loads((R/'diagnostics/calib_final_source.json').read_text())['cam1'];S=mask_head_scale_features(h,g,cal);D=density_features(r,h,g);E=boundary_features(h,g)
    return [np.column_stack([r[:,:39],h[:,:126],S,D,E,original,original-g]),np.column_stack([r[:,39:231],h[:,126:]]),np.column_stack([r[:,231:],c,crop])]

class RecoveredClassifier:
    def __init__(self,path=None,device='cpu'):
        # Only locally produced, trusted checkpoints are supported.
        cp=torch.load(path or R/'recovered_classifier.pt',map_location='cpu',weights_only=False);self.device=device;self.model=ScaleExperts(cp['dims']).eval().to(device);self.model.load_state_dict(cp['state_dict']);self.means=cp['means'];self.stds=cp['stds'];self.helper=InsideOutside.__new__(InsideOutside);self.helper._prepare_floor()
    def predict_features(self,A,geometry,original_geometry):
        parts=feature_parts(A,geometry,original_geometry);X=[torch.from_numpy(((p-m)/s).astype('float32')).to(self.device) for p,m,s in zip(parts,self.means,self.stds)]
        with torch.inference_mode():return self.model(*X)[0].softmax(1).cpu().numpy()
    def predict_prepared(self,full,crop,geometry,original_geometry):
        A=np.concatenate([expanded_features(full,crop[:,:,3],geometry,self.helper.signed),crop_features(crop)]);p=self.predict_features(A,geometry,original_geometry)[0];return self._answer(p)
    @staticmethod
    def _answer(p):
        names=('outside','inside','doorway');i=int(p.argmax());return {'label':names[i],'confidence':float(p[i]),'probabilities':dict(zip(names,map(float,p)))}
    def prepare_rgb(self,frame,person_mask,*,box=None,foot=None,floor_px=None):
        if frame.dtype!=np.uint8 or frame.ndim!=3 or frame.shape[2]!=3 or person_mask.shape!=frame.shape[:2]:raise ValueError('Expected RGB uint8 and same-size target mask')
        h,w=person_mask.shape;sx,sy=1280/w,720/h;rgb=cv2.resize(frame,(1280,720));mask=cv2.resize((person_mask!=0).astype('uint8')*255,(1280,720),interpolation=cv2.INTER_NEAREST);rr,cc=np.where(mask)
        if not len(rr):raise ValueError('Empty target mask')
        x1,y1,x2,y2=np.asarray(box)*[sx,sy,sx,sy] if box is not None else [cc.min(),rr.min(),cc.max()+1,rr.max()+1]
        if foot is None:
            _,labels,stats,_=cv2.connectedComponentsWithStats(mask);br,bc=np.where(labels==(1+np.argmax(stats[1:,cv2.CC_STAT_AREA])));fy=float(br.max());fx=float(bc[br>=fy-2].mean())
        else:fx,fy=np.asarray(foot)*[sx,sy]
        if floor_px is None:floor_px=round(float(self.helper.floor_distance[int(np.clip(fy,0,719)),int(np.clip(fx,0,1279))]),1)
        original=np.array([x1/1280,y1/720,x2/1280,y2/720,(x2-x1)/1280,(y2-y1)/720,fx/1280,fy/720,floor_px/100],np.float32)
        full=np.dstack([cv2.resize(rgb.astype('float32'),(320,180),interpolation=cv2.INTER_AREA).astype('uint8'),cv2.resize(mask.astype('float32'),(320,180),interpolation=cv2.INTER_AREA).astype('uint8'),self.helper.small_floor]);r,c=np.where(full[:,:,3]>63)
        if not len(r):raise ValueError('Target too small after resize')
        bounds=np.array([c.min()/320,r.min()/180,(c.max()+1)/320,(r.max()+1)/180],dtype=float);g=original.copy();g[:4]=bounds;g[4:6]=g[2:4]-g[:2];x1,y1,x2,y2=bounds*[1280,720,1280,720];bw=x2-x1;bh=y2-y1;a,b=max(0,int(x1-bw/2)),max(0,int(y1-bh/8));c,d=min(1280,int(np.ceil(x2+bw/2))),min(720,int(np.ceil(y2+bh/8)))
        source=np.dstack([rgb,mask,self.helper.floor]).astype('float32');crop=np.dstack([cv2.resize(source[b:d,a:c,:4],(96,160),interpolation=cv2.INTER_AREA),cv2.resize(source[b:d,a:c,4],(96,160),interpolation=cv2.INTER_AREA)]).astype('uint8')
        return full,crop,g,original
    def predict_rgb(self,frame,person_mask,**kwargs):return self.predict_prepared(*self.prepare_rgb(frame,person_mask,**kwargs))
