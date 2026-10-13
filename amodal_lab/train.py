"""Tiny RGB+visible-mask completion experiment. Only owner-approved full/good masks.
No real hidden-mask ground truth: evaluation measures synthetic completion only.
"""
import argparse,json,random,hashlib
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.utils.data import Dataset,DataLoader
ROOT=Path(__file__).resolve().parent

def occlude(rgb,mask,rng):
    h,w=mask.shape;occ=np.zeros_like(mask,dtype=bool)
    ys,xs=np.where(mask>0)
    if not len(xs):raise ValueError('Empty target mask')
    # Retry so training never labels an unoccluded image as occluded.
    for _ in range(30):
        occ[:]=False;kind=int(rng.integers(0,4))
        if kind==0: # vertical pole
            x=int(rng.choice(xs));bw=int(rng.integers(max(2,w//12),max(3,w//3)));occ[:,max(0,x-bw//2):min(w,x+bw//2+1)]=True
        elif kind==1: # stand, covers legs
            y=int(rng.integers(int(ys.min()),int(ys.max())+1));occ[y:,:]=True
        elif kind==2: # upper occluder
            y=int(rng.integers(int(ys.min()),int(ys.max())+1));occ[:y,:]=True
        else: # approximate other person's silhouette
            yy,xx=np.mgrid[:h,:w];x=int(rng.choice(xs));y=int(rng.choice(ys));occ=((xx-x)/(w*.22))**2+((yy-y)/(h*.45))**2<1
        frac=(occ & (mask>0)).sum()/max(1,mask.sum())
        if .1<frac<.75:break
    if not .1<frac<.75:
        occ[:]=False;occ[:,int(np.median(xs)):]=True
    visible=mask*(~occ);rgb=rgb.copy();rgb[occ]=rng.uniform(.1,.8,size=3)
    return rgb,visible,occ

class Crops(Dataset):
    def __init__(self,items,task,train,augment="legacy",repeat=1,input_mode="rgb_mask"):self.items=items;self.task=task;self.train=train;self.augment=augment;self.repeat=repeat;self.input_mode=input_mode
    def __len__(self):return len(self.items)*self.repeat
    def __getitem__(self,i):
        row=self.items[i%len(self.items)];rgb=np.asarray(Image.open(ROOT/'data'/row['image']).convert('RGB').resize((96,160)),dtype=np.float32)/255
        m=(np.asarray(Image.open(ROOT/'data'/row['label'].get('mask_path',row['mask'])).convert('L').resize((96,160),Image.Resampling.NEAREST))>127).astype(np.float32)
        if self.task=='visibility':return torch.from_numpy(m[None] if self.input_mode=='mask' else np.concatenate([rgb,m[:,:,None]],2).transpose(2,0,1)),torch.tensor(float(row['label']['visibility']=='occluded'))
        seed=int(hashlib.sha256(row['id'].encode()).hexdigest()[:8],16)
        rng=np.random.default_rng(int(np.random.randint(0,2**31))) if self.train else np.random.default_rng(seed)
        if self.train and self.augment=="rich":
            from augment import augment_pair,rich_occlude
            rgb,m=augment_pair(rgb,m,rng);rgb,v,o=rich_occlude(rgb,m,rng)
        else:rgb,v,o=occlude(rgb,m,rng)
        return torch.from_numpy(v[None] if self.input_mode=="mask" else np.concatenate([rgb,v[:,:,None]],2).transpose(2,0,1)),torch.from_numpy(m[None]),torch.from_numpy(o[None]),torch.from_numpy(v[None])

def block(a,b):return nn.Sequential(nn.Conv2d(a,b,3,padding=1),nn.ReLU(),nn.Conv2d(b,b,3,padding=1),nn.ReLU())
class Tiny(nn.Module):
    def __init__(self,task,input_channels=4,spatial=False):
        super().__init__();self.task=task;self.spatial=spatial;self.a=block(input_channels,16);self.b=block(16,32);self.c=block(32,64);self.d=block(96,32);self.e=block(48,16);self.out=nn.Conv2d(16,1,1);self.cls=nn.Linear(64*15 if spatial else 64,1)
    def forward(self,x):
        a=self.a(x);b=self.b(nn.functional.max_pool2d(a,2));c=self.c(nn.functional.max_pool2d(b,2))
        if self.task=='visibility':return self.cls(nn.functional.adaptive_avg_pool2d(c,(5,3)).flatten(1) if self.spatial else c.mean((2,3))).squeeze(1)
        d=self.d(torch.cat([nn.functional.interpolate(c,size=b.shape[2:],mode='nearest'),b],1));e=self.e(torch.cat([nn.functional.interpolate(d,size=a.shape[2:],mode='nearest'),a],1));return self.out(e)

def main():
    p=argparse.ArgumentParser();p.add_argument('--task',choices=['completion','visibility'],default='completion');p.add_argument('--epochs',type=int,default=20);p.add_argument('--batch',type=int,default=8);p.add_argument('--val-day',required=True);p.add_argument('--device',default='mps' if torch.backends.mps.is_available() else 'cpu');p.add_argument("--augment",choices=["legacy","rich"],default="legacy");p.add_argument("--repeat",type=int,default=1);p.add_argument("--run-name");p.add_argument("--labels");p.add_argument("--spatial",action="store_true");p.add_argument("--balanced",action="store_true");p.add_argument("--input-mode",choices=["rgb_mask","mask"],default="rgb_mask");args=p.parse_args()
    torch.manual_seed(42);random.seed(42);np.random.seed(42)
    label_path=Path(args.labels) if args.labels else ROOT/'data/labels.json'
    label_bytes=label_path.read_bytes();labels=json.loads(label_bytes);items=json.loads((ROOT/'data/manifest.json').read_text())['items'];eligible=[]
    for r in items:
        l=labels.get(r['id'],{})
        if l.get('quality')!='good':continue
        if l.get('visibility') not in (['full'] if args.task=='completion' else ['full','occluded']):continue
        target=ROOT/'data'/l.get('mask_path',r['mask'])
        if not target.exists() or not np.asarray(Image.open(target).convert('L')).any():
            raise SystemExit(f'Нет допустимой маски: {r["id"]}')
        eligible.append(dict(r,label=l))
    tr=[r for r in eligible if r['day']!=args.val_day];va=[r for r in eligible if r['day']==args.val_day]
    if len(tr)<20 or len(va)<5:raise SystemExit(f'Нужно минимум 20 подходящих train и 5 val. Сейчас {len(tr)} / {len(va)}. Разметь полные хорошие маски на разных днях.')
    if args.task=='visibility' and any(len({r['label']['visibility'] for r in s})<2 for s in [tr,va]):raise SystemExit('Для видимости нужны оба класса в train и val.')
    out=ROOT/'runs'/(args.run_name or args.task);out.mkdir(parents=True,exist_ok=True)
    config=dict(vars(args),train_ids=[r['id'] for r in tr],val_ids=[r['id'] for r in va],evaluation='synthetic occlusions only' if args.task=='completion' else 'owner visibility labels',limitation='split by day, cross-day identities unverified',mask_sources={r['id']:dict(path=r['label'].get('mask_path',r['mask']),sha256=hashlib.sha256((ROOT/'data'/r['label'].get('mask_path',r['mask'])).read_bytes()).hexdigest()) for r in eligible},labels_sha256=hashlib.sha256(label_bytes).hexdigest())
    (out/'labels_snapshot.json').write_bytes(label_bytes)
    (out/'split.json').write_text(json.dumps(config,indent=2))
    model=Tiny(args.task,1 if args.input_mode=="mask" else 4,args.spatial).to(args.device);opt=torch.optim.Adam(model.parameters(),lr=.001);lossfn=nn.BCEWithLogitsLoss(pos_weight=torch.tensor(sum(r["label"]["visibility"]=="full" for r in tr)/max(1,sum(r["label"]["visibility"]=="occluded" for r in tr)),device=args.device) if args.balanced and args.task=="visibility" else None);best=-1;history=[]
    train=DataLoader(Crops(tr,args.task,True,args.augment,args.repeat,args.input_mode),batch_size=args.batch,shuffle=True);val=DataLoader(Crops(va,args.task,False,input_mode=args.input_mode),batch_size=args.batch)
    for epoch in range(args.epochs):
        model.train();losses=[]
        for batch in train:
            x,y=batch[:2];x,y=x.to(args.device),y.to(args.device);pred=model(x);loss=lossfn(pred,y)
            if args.task=='completion':
                prob=pred.sigmoid();loss=loss+1-(2*(prob*y).sum()+1)/(prob.sum()+y.sum()+1)
            opt.zero_grad();loss.backward();opt.step();losses.append(loss.item())
        model.eval();inter=union=hidden_i=hidden_u=correct=n=0;recalls=[[],[]]
        with torch.no_grad():
            for batch in val:
                x,y=batch[:2];pred=(model(x.to(args.device)).cpu().sigmoid()>.5)
                if args.task=='visibility':
                    correct+=(pred==y.bool()).sum().item();n+=len(y)
                    for cls in [0,1]:recalls[cls].extend((pred[y==cls]==cls).tolist())
                else:
                    occ,v=batch[2:];pred=pred|v.bool();truth=y.bool();inter+=(pred&truth).sum().item();union+=(pred|truth).sum().item();hidden_i+=(pred&truth&occ).sum().item();hidden_u+=((pred|truth)&occ).sum().item()
        score=(sum(np.mean(a) for a in recalls)/2) if args.task=='visibility' else hidden_i/max(1,hidden_u)
        report=dict(epoch=epoch+1,loss=float(np.mean(losses)),score=float(score),metric='balanced_accuracy' if args.task=='visibility' else 'hidden_region_IoU',full_mask_IoU=inter/max(1,union) if args.task=='completion' else None)
        history.append(report);(out/'history.json').write_text(json.dumps(history,indent=2));print(json.dumps(report),flush=True)
        if score>best:
            best=score;torch.save(dict(model=model.cpu().state_dict(),config=config,report=report),out/'best.pt');model.to(args.device)
if __name__=='__main__':main()
