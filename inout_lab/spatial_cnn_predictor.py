"""Run a saved spatial-channel CNN on camera 1 RGB + person mask.
Weights in diagnostics are day-excluded experiments, not full-data production
models. Spatial priors must use the exact same excluded day as the checkpoint.
"""
from pathlib import Path
import pickle
import cv2
import numpy as np
import torch
from torch.nn import functional as F
from inside_outside import InsideOutside
from spatial_cnn import SpatialNet
from door_features import expanded_features

class SpatialCNNPredictor(InsideOutside):
    def __init__(self,checkpoint,prior_path=None,device=None,threshold=.9):
        path=Path(checkpoint)
        data=torch.load(path,map_location='cpu',weights_only=False)
        self.mode=data['mode'];self.device=device or ('mps' if torch.backends.mps.is_available() else 'cpu');self.threshold=threshold
        prior_path=prior_path or path.parent/('spatial_cnn_prior_excluding_'+data['day_excluded']+'.pkl')
        with open(prior_path,'rb') as f:self.prior=pickle.load(f)
        self.geometry_mean=np.asarray(data['geometry_mean'],np.float32);self.geometry_std=np.asarray(data['geometry_std'],np.float32)
        channels=data['state_dict']['blocks.0.weight'].shape[1]
        self.model=SpatialNet(channels,len(self.geometry_mean)).to(self.device)
        self.model.load_state_dict(data['state_dict']);self.model.eval()
        self._prepare_floor()
        xx,yy=np.meshgrid(np.linspace(0,255,1280),np.linspace(0,255,720))
        self.coordinates=cv2.resize(np.dstack([xx,yy]).astype(np.float32),(320,180),interpolation=cv2.INTER_AREA).astype(np.uint8)
        self.map_tensor=torch.from_numpy(self.prior.channels((88,160)).transpose(2,0,1).copy())[None]*255

    def _features(self,full,crop_mask,geometry,signed=None):
        if full.shape[2]==5:full=np.dstack([full,self.coordinates])
        image=F.interpolate(torch.from_numpy(full.transpose(2,0,1).copy()).float()[None],size=(88,160),mode='bilinear',align_corners=False)
        if self.mode!='baseline':image=torch.cat([image,self.map_tensor],1)
        g=np.asarray(geometry,np.float32)
        if self.mode=='maps_knn':
            a=expanded_features(full,crop_mask,geometry,signed)
            g=np.concatenate([g,self.prior.transform(a[None])[0]])
        g=(g-self.geometry_mean)/self.geometry_std
        return image,torch.from_numpy(g)[None]

    def _predict(self,prepared):
        image,g=prepared
        with torch.inference_mode():p=self.model(image.to(self.device),g.to(self.device)).softmax(1).cpu().numpy()[0]
        i=int(p.argmax());names=('outside','inside','doorway');uncertain=bool(p[i]<self.threshold)
        return dict(label='unknown' if uncertain else names[i],raw_label=names[i],confidence=float(p[i]),uncertain=uncertain,probabilities={name:float(value) for name,value in zip(names,p)})
