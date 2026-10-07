"""Camera1 dual-view CNN API; diagnostic day-excluded checkpoints are explicit.
Loads only locally generated trusted checkpoints; no downloads or installations.
"""
from pathlib import Path
import numpy as np,cv2,torch
from torch.nn import functional as F
from inside_outside import InsideOutside
from dual_view_cnn import DualViewNet
class DualViewPredictor(InsideOutside):
 requires_crop_rgb=True
 def __init__(self,checkpoint,device=None):
  data=torch.load(Path(checkpoint),map_location='cpu',weights_only=False);self.held_day=data['held_day'];self.device=device or ('mps' if torch.backends.mps.is_available() else 'cpu');self.geometry_mean=np.asarray(data['geometry_mean'],np.float32);self.geometry_std=np.asarray(data['geometry_std'],np.float32);self.model=DualViewNet().to(self.device);self.model.load_state_dict(data['state_dict']);self.model.eval();self._prepare_floor();xx,yy=np.meshgrid(np.linspace(0,255,1280),np.linspace(0,255,720));self.coordinate_full=np.dstack([xx,yy]).astype(np.float32);self.coordinate_small=cv2.resize(self.coordinate_full,(320,180),interpolation=cv2.INTER_AREA).astype(np.uint8)
 def predict_prepared(self,full,crop,geometry):
  if full.shape!=(180,320,7) or crop.shape!=(160,96,7):raise ValueError('Expected full180x320x7 and crop160x96x7, RGB+mask+floor+x/y')
  g=np.asarray(geometry,np.float32)
  if g.shape!=(9,) or not np.isfinite(g).all():raise ValueError('Expected nine finite geometry values')
  if not np.isfinite(full).all() or not np.isfinite(crop).all():raise ValueError('Non-finite image channels')
  if not np.any(full[:,:,3]):raise ValueError('Empty target mask')
  images=[]
  for view in [full,crop]:
   tensor=torch.from_numpy(view.transpose(2,0,1).copy()).float()[None].to(self.device);images.append(F.interpolate(tensor,size=(88,160),mode='bilinear',align_corners=False))
  gt=torch.from_numpy((g-self.geometry_mean)/self.geometry_std)[None].to(self.device)
  with torch.inference_mode():p=self.model(*images,gt).softmax(1).cpu().numpy()[0]
  names=('outside','inside','doorway');i=int(p.argmax());return {'label':names[i],'confidence':float(p[i]),'probabilities':dict(zip(names,map(float,p))),'held_day':self.held_day}
 def _predict_inputs(self,full,crop_mask,geometry,signed=None,crop_rgb=None,crop_floor=None):
  if crop_rgb is None or crop_floor is None:raise ValueError('Detailed RGB/floor crop required')
  g=np.asarray(geometry);x1,y1,x2,y2=g[:4]*[1280,720,1280,720];a,b=max(0,int(x1-(x2-x1)/2)),max(0,int(y1-(y2-y1)*.125));c,d=min(1280,int(np.ceil(x2+(x2-x1)/2))),min(720,int(np.ceil(y2+(y2-y1)*.125)))
  xy=cv2.resize(self.coordinate_full[b:d,a:c],(96,160),interpolation=cv2.INTER_AREA).astype(np.uint8)
  return self.predict_prepared(np.dstack([full,self.coordinate_small]),np.dstack([crop_rgb,crop_mask,crop_floor,xy]),geometry)
