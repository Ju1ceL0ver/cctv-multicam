"""64x64 aligned RGB/mask/floor/coordinates + cheap physical-scale features.
Three conv64/128/256 blocks; FC256/64/3. All weights train jointly on MPS.
Checkpoint every completed held-day evaluation; no unknown/doorway exclusions.
"""
from pathlib import Path
import argparse,json,time
import cv2,numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from align_visible_bbox import align_sample
from mask_density_features import density_features
from head_scale_search import mask_head_scale_features

ROOT=Path(__file__).parent

class Tiny(nn.Module):
    def __init__(self,geometry_dim):
        super().__init__();layers=[];previous=7
        for channels in [64,128,256]:
            layers.extend([nn.Conv2d(previous,channels,3,stride=2,padding=1),nn.BatchNorm2d(channels),nn.ReLU(),nn.MaxPool2d(2)]);previous=channels
        self.image=nn.Sequential(*layers);self.geometry=nn.Sequential(nn.Linear(geometry_dim,64),nn.ReLU())
        self.classifier=nn.Sequential(nn.Linear(320,256),nn.ReLU(),nn.Dropout(.2),nn.Linear(256,64),nn.ReLU(),nn.Linear(64,3))
    def forward(self,image,geometry):
        return self.classifier(torch.cat([self.image(image).flatten(1),self.geometry(geometry)],1))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--resume',action='store_true');args=parser.parse_args()
    start=time.perf_counter();z=np.load(ROOT/'io_cam1.npz');y=z['y'];days=z['day'];valid=np.load(ROOT/'diagnostics/original_labels.npy')!=0
    image_path=ROOT/'diagnostics/aligned_cnn_64.npy'
    if not image_path.exists():
        X=z['X'];F0=z['F'];rawG=z['G'][:,1:];images=[]
        for f,x,g in zip(F0,X,rawG):
            crop,_,_=align_sample(f,x,g);images.append(np.stack([cv2.resize(crop[:,:,j],(64,64),interpolation=cv2.INTER_AREA) for j in range(7)],2))
        np.save(image_path,np.asarray(images,np.uint8));del X,F0,images
    images=np.load(image_path,mmap_mode='r');A=np.load(ROOT/'diagnostics/aligned_features.npy');G=np.load(ROOT/'diagnostics/aligned_geometry.npy');cal=json.loads((ROOT/'diagnostics/calib_final_source.json').read_text())['cam1']
    geo=np.column_stack([G,density_features(A[:,:234],A[:,234:936],G),mask_head_scale_features(A[:,234:936],G,cal)]).astype(np.float32)
    device='mps' if torch.backends.mps.is_available() else 'cpu';torch.set_num_threads(2);P=np.zeros((len(y),3));folds={}
    if args.resume:
        progress=json.loads((ROOT/'aligned_tiny_cnn_progress.json').read_text());saved=np.load(ROOT/'diagnostics/aligned_tiny_cnn_predictions.npz');folds=progress['folds'];P=saved['probabilities']
        assert set(saved['completed_days'].tolist())==set(folds), 'Progress/prediction checkpoint mismatch'
        covered=np.isin(days,list(folds));assert np.isfinite(P[covered]).all() and np.max(np.abs(P[covered].sum(1)-1))<1e-5
        print('resume completed days',list(folds),flush=True)
    print('tiny aligned CNN prepared',len(images),'device',device,'seconds',round(time.perf_counter()-start),flush=True)
    for day in np.unique(days):
        if str(day) in folds:continue
        tr=np.where(valid&(days!=day))[0];te=np.where(days==day)[0];torch.manual_seed(19)
        mu=geo[tr].mean(0);std=np.maximum(geo[tr].std(0),.01)
        train=torch.from_numpy(np.asarray(images[tr]).transpose(0,3,1,2).astype(np.float32)/255).to(device);test=torch.from_numpy(np.asarray(images[te]).transpose(0,3,1,2).astype(np.float32)/255).to(device)
        trainG=torch.from_numpy((geo[tr]-mu)/std).to(device);testG=torch.from_numpy((geo[te]-mu)/std).to(device);target=torch.from_numpy(y[tr]).long().to(device)
        model=Tiny(geo.shape[1]).to(device);opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.002);schedule=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=35)
        weights=torch.tensor(np.sqrt(len(tr)/(3*np.bincount(y[tr],minlength=3))),dtype=torch.float32,device=device)
        for epoch in range(35):
            model.train()
            for idx in torch.randperm(len(tr)).split(128):
                image=train[idx].clone();t=target[idx];b=len(idx)
                gain=torch.rand((b,1,1,1),device=device)*.3+.85;offset=(torch.rand((b,1,1,1),device=device)-.5)*.08
                image[:,:3]=(image[:,:3]*gain+offset+torch.randn_like(image[:,:3])*.01).clamp(0,1)
                flip=torch.rand(b,device=device)<.5;image[flip]=image[flip].flip(-1)
                logits=model(image,trainG[idx]);prob=logits.softmax(1);definite=t<2
                opposite=prob[torch.arange(b,device=device),1-t.clamp(max=1)]
                loss=F.cross_entropy(logits,t,weight=weights)+2*(opposite*definite).mean();opt.zero_grad(set_to_none=True);loss.backward();opt.step()
            schedule.step()
        model.eval()
        with torch.inference_mode():P[te]=model(test,testG).softmax(1).cpu().numpy()
        pred=P[te].argmax(1);v=valid[te];gross=v&(y[te]<2)&(pred<2)&(pred!=y[te]);folds[str(day)]={'wrong':int((v&(pred!=y[te])).sum()),'gross':int(gross.sum()),'gross_indices':te[gross].tolist()}
        np.savez(ROOT/'diagnostics/aligned_tiny_cnn_predictions.npz',probabilities=P,completed_days=np.array(list(folds)))
        if str(day)=='20260919':torch.save({'state_dict':{k:v.detach().cpu() for k,v in model.state_dict().items()},'geometry_mean':mu,'geometry_std':std,'held_day':str(day),'geometry_dim':geo.shape[1]},ROOT/'diagnostics/aligned_tiny_20260919.pt')
        (ROOT/'aligned_tiny_cnn_progress.json').write_text(json.dumps({'folds':folds,'seconds':time.perf_counter()-start},indent=2));print('tiny CNN',day,folds[str(day)],'seconds',round(time.perf_counter()-start),flush=True)
        del model,opt,train,test,trainG,testG
        if device=='mps':torch.mps.empty_cache()
    pred=P.argmax(1);gross=valid&(y<2)&(pred<2)&(pred!=y);report={'defined':int(valid.sum()),'accuracy':float((pred[valid]==y[valid]).mean()),'wrong':int((valid&(pred!=y)).sum()),'gross':int(gross.sum()),'gross_indices':np.where(gross)[0].tolist(),'doorway_recall':float((pred[y==2]==2).mean()),'folds':folds,'seconds':time.perf_counter()-start}
    (ROOT/'aligned_tiny_cnn_report.json').write_text(json.dumps(report,indent=2));print('TINY CNN RESULT',json.dumps(report),flush=True)
