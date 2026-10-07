"""Existing local YOLO26n-pose backbone, frozen mask-conditioned features.
No downloads/installation/training and no manual labels consumed.
"""
from pathlib import Path
import os,json,time,hashlib,zipfile,shutil,tempfile,argparse
R=Path(__file__).parent
os.environ['YOLO_CONFIG_DIR']=str(R/'diagnostics/yolo_config')
import cv2,numpy as np,torch
from torch.nn import functional as F
from ultralytics import YOLO
from align_visible_bbox import align_sample

class BodyFeatures(torch.nn.Module):
    def __init__(self):
        super().__init__();model=YOLO(str(R/'weights/yolo26n-pose.pt')).model
        self.blocks=torch.nn.ModuleList(list(model.model.children())[:11]);assert all(b.f==-1 for b in self.blocks)
        for p in self.parameters():p.requires_grad_(False)
    def forward(self,rgb,mask):
        x=rgb;out=[]
        for j,b in enumerate(self.blocks):
            x=b(x)
            if j not in [4,6,10]:continue
            m=F.interpolate(mask,size=x.shape[2:],mode='area');den=m.sum((2,3)).clamp(min=.001)
            out.extend([(x*m).sum((2,3))/den,(x*(1-m)).sum((2,3))/(1-m).sum((2,3)).clamp(min=.001),F.adaptive_avg_pool2d(x,(2,2)).flatten(1)])
        return torch.cat(out,1)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--source-crops');ap.add_argument('--tag',default='');args=ap.parse_args();suffix=('_'+args.tag) if args.tag else ''
    torch.set_num_threads(1);start=time.perf_counter();model=BodyFeatures().eval();z=np.load(R/'io_cam1.npz');g=z['G'][:,1:];rows=[]
    temporary=tempfile.TemporaryDirectory(prefix='body_features_',dir=R/'diagnostics')
    source=Path(args.source_crops) if args.source_crops else R/'io_cam1.npz'
    if args.source_crops:
        recovered=np.load(source);assert np.array_equal(recovered['ids'],z['ids']) and np.array_equal(recovered['y'],z['y']) and np.array_equal(recovered['day'],z['day'])
    with zipfile.ZipFile(source) as archive:
        for key in (['X'] if args.source_crops else ['F','X']):
            with archive.open(key+'.npy') as src,open(Path(temporary.name)/(key+'.npy'),'wb') as dst:shutil.copyfileobj(src,dst)
    X=np.load(Path(temporary.name)/'X.npy',mmap_mode='r');full=None if args.source_crops else np.load(Path(temporary.name)/'F.npy',mmap_mode='r')
    for start_i in range(0,len(X),16):
        crops=[X[i] if args.source_crops else align_sample(full[i],X[i],g[i])[0] for i in range(start_i,min(start_i+16,len(X)))];a=np.stack(crops);rgb=torch.from_numpy(a[:,:,:,:3].transpose(0,3,1,2).copy()).float()/255;mask=torch.from_numpy(a[:,:,:,3:4].transpose(0,3,1,2).copy()).float()/255
        with torch.inference_mode():out=model(rgb,mask).numpy()
        rows.append(out)
        if start_i%256==0:print('frozen body',start_i+len(a),'elapsed',round(time.perf_counter()-start),flush=True)
    B=np.vstack(rows);assert len(B)==1628 and np.isfinite(B).all();np.save(R/f'diagnostics/frozen_body{suffix}_features.npy',B)
    report={'shape':list(B.shape),'seconds':time.perf_counter()-start,'labels_used':False,'training':False,'parameters':sum(p.numel() for p in model.parameters()),'weights_sha256':hashlib.sha256((R/'weights/yolo26n-pose.pt').read_bytes()).hexdigest(),'input':'aligned crop RGB 160x96 plus target mask; YOLO frozen backbone blocks0-10','new_downloads':False,'source_crops':args.source_crops};(R/f'frozen_body{suffix}_features_report.json').write_text(json.dumps(report,indent=2));print(report,flush=True)

    del X,full
    temporary.cleanup()
