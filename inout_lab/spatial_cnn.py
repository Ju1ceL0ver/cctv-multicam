"""Actual dense spatial-prior channels on MPS; hard-day diagnostic ablation.
Training priors cross-fitted by day; held-day labels never enter any prior/map.
"""
from pathlib import Path
import argparse,json,time,pickle
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from spatial_prior import SpatialPrior
root=Path(__file__).parent

class SpatialNet(nn.Module):
    def __init__(self,channels,geometry):
        super().__init__();self.blocks=nn.Sequential(nn.Conv2d(channels,64,3,padding=1),nn.ReLU(),nn.MaxPool2d(2),nn.Conv2d(64,128,3,padding=1),nn.ReLU(),nn.MaxPool2d(2),nn.Conv2d(128,256,3,padding=1),nn.ReLU(),nn.MaxPool2d(2));self.head=nn.Sequential(nn.Linear(256*6+geometry,256),nn.ReLU(),nn.Linear(256,64),nn.ReLU(),nn.Linear(64,3))
    def forward(self,image,geometry):
        x=self.blocks(image/127.5-1)
        grid=F.adaptive_avg_pool2d(F.interpolate(x,size=(10,20),mode='bilinear',align_corners=False),(2,2)).flatten(1)
        mask=F.interpolate(image[:,3:4]/255.,size=x.shape[-2:],mode='area')
        pool=(x*mask).sum((2,3))/mask.sum((2,3)).clamp_min(.01)
        # Visible upper-body pool, independent of anatomical feet.
        rows=torch.arange(mask.shape[-2],device=x.device)[None,None,:,None]
        mass=mask.sum(3,keepdim=True);upper=(mass.cumsum(2)<=mass.sum(2,keepdim=True)*.3).to(mask.dtype)*mask
        top=(x*upper).sum((2,3))/upper.sum((2,3)).clamp_min(.01)
        return self.head(torch.cat([grid,pool,top,geometry],1))


def cached_frames(z):
    """Exact float32 Torch resize, file-backed; keep only each minibatch on MPS."""
    path=root/'cnn_frames_88x160.npy';meta=root/'cnn_frames_88x160.json'
    source=root/'io_cam1.npz';signature={'size':source.stat().st_size,'mtime_ns':source.stat().st_mtime_ns,'size_hw':[88,160]}
    if not path.exists() or not meta.exists() or json.loads(meta.read_text())!=signature:
        raw=z['F'];out=np.lib.format.open_memmap(path,mode='w+',dtype=np.float32,shape=(len(raw),7,88,160))
        for start in range(0,len(raw),16):
            chunk=torch.from_numpy(raw[start:start+16].transpose(0,3,1,2).copy()).float()
            out[start:start+16]=F.interpolate(chunk,size=(88,160),mode='bilinear',align_corners=False).numpy()
        out.flush();del raw,out;meta.write_text(json.dumps(signature))
    return np.load(path,mmap_mode='r')

def run(day,epochs=30,modes=('baseline','maps','maps_knn')):
    z=np.load(root/'io_cam1.npz');raw=np.load(root/'diagnostics/original_labels.npy');y=z['y'];days=z['day'];valid=raw!=0;A=np.column_stack([np.load(root/('diagnostics/'+n+'_features.npy')) for n in ['rich','head','context']]);G=z['G'][:,1:];tr=np.where(valid&(days!=day))[0];te=np.where(valid&(days==day))[0];whole=cached_frames(z);device='mps' if torch.backends.mps.is_available() else 'cpu'
    prior=SpatialPrior().fit(A[tr],y[tr]);(root/('diagnostics/spatial_cnn_prior_excluding_'+day+'.pkl')).write_bytes(pickle.dumps(prior,protocol=5));map_bank=[prior.channels((88,160)).transpose(2,0,1)*255];map_index=np.zeros(len(y),np.int64);extras=np.zeros((len(y),144),np.float32)
    extras[te]=prior.transform(A[te])
    for cal in np.unique(days[tr]):
        fit=valid&~np.isin(days,[day,cal]);ix=tr[days[tr]==cal];local=SpatialPrior().fit(A[fit],y[fit]);map_index[ix]=len(map_bank);map_bank.append(local.channels((88,160)).transpose(2,0,1)*255);extras[ix]=local.transform(A[ix])
    report={};all_prob={}
    for mode in modes:
        torch.manual_seed(0);geometry=np.column_stack([G,extras]) if mode=='maps_knn' else G;gm=geometry[tr].mean(0);gs=np.maximum(geometry[tr].std(0),.01)
        channels=7 if mode=='baseline' else 10
        def get_image(indices):
            data=np.array(whole[indices],copy=True)
            if mode!='baseline':data=np.concatenate([data,np.asarray(map_bank)[map_index[indices]]],axis=1)
            return torch.from_numpy(data).to(device)
        train_g=torch.from_numpy(((geometry[tr]-gm)/gs).astype(np.float32)).to(device);target=torch.from_numpy(y[tr]).long().to(device);test_g=torch.from_numpy(((geometry[te]-gm)/gs).astype(np.float32)).to(device)
        net=SpatialNet(channels,geometry.shape[1]).to(device);opt=torch.optim.AdamW(net.parameters(),lr=.001,weight_decay=.0001);weights=torch.tensor(np.sqrt(len(tr)/(3*np.bincount(y[tr],minlength=3))),dtype=torch.float32,device=device);start=time.perf_counter()
        for epoch in range(epochs):
            net.train();correct=0;losses=[]
            for idx in torch.randperm(len(tr)).split(64):
                f=get_image(tr[idx.numpy()]);f[:,:3]=(f[:,:3]*torch.empty(len(idx),1,1,1,device=device).uniform_(.9,1.1)+torch.randn_like(f[:,:3])*1.5).clamp(0,255);p=net(f,train_g[idx]);loss=F.cross_entropy(p,target[idx],weight=weights);opt.zero_grad(set_to_none=True);loss.backward();opt.step();correct+=(p.detach().argmax(1)==target[idx]).sum();losses.append(loss.detach())
            if (epoch+1)%5==0:print(day,mode,'epoch',epoch+1,'train',float(correct)/len(tr),'seconds',round(time.perf_counter()-start),flush=True)
        net.eval()
        with torch.inference_mode():p=torch.cat([net(get_image(te[b.numpy()]),test_g[b]).softmax(1).cpu() for b in torch.arange(len(te)).split(64)]).numpy()
        pred=p.argmax(1);truth=y[te];gross=(truth<2)&(pred<2)&(truth!=pred);report[mode]={'accuracy':float((pred==truth).mean()),'wrong':int((pred!=truth).sum()),'gross':int(gross.sum()),'n':len(te),'params':sum(v.numel() for v in net.parameters()),'seconds':time.perf_counter()-start};all_prob[mode]=p;print('RESULT',day,mode,report[mode],flush=True)
        torch.save({'state_dict':net.cpu().state_dict(),'geometry_mean':gm,'geometry_std':gs,'mode':mode,'day_excluded':day},root/('diagnostics/spatial_cnn_'+day+'_'+mode+'.pt'))
        del train_g,test_g,net,opt
        if device=='mps':torch.mps.empty_cache()
    (root/('spatial_cnn_'+day+'_report.json')).write_text(json.dumps(report,indent=2));np.savez(root/('diagnostics/spatial_cnn_'+day+'.npz'),indices=te,**all_prob)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--day',default='20260918');p.add_argument('--epochs',type=int,default=30);args=p.parse_args();run(args.day,args.epochs)
