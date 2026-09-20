"""Train one candidate per reviewed dataset version; never replace production weights."""
import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from storage import atomic_json, file_lock, read_json
from export_seg_dataset import source_manifest, export
ROOT=Path(__file__).resolve().parent
MODELS=ROOT.parent/'retail_analytics/models'
STATE=ROOT/'data/student_training_state.json'


def should_train(now=None):
    now=now or datetime.now()
    manifest=source_manifest()
    if not manifest['sources']:return False
    previous=read_json(STATE,{})
    if previous.get('dataset_id')!=manifest['id']:return True
    if previous.get('status')=='complete':return False
    # At most one failed/interrupted attempt on an unchanged dataset per night.
    return previous.get('attempt_day')!=now.strftime('%Y%m%d')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',default=str(MODELS/'yolo26n-seg.pt'));p.add_argument('--imgsz',type=int,default=960)
    p.add_argument('--epochs',type=int,default=80);p.add_argument('--batch',type=int,default=8)
    p.add_argument('--export-only',action='store_true');p.add_argument('--force',action='store_true')
    args=p.parse_args()
    with file_lock(ROOT/'data/.student_training.lock',timeout=1):
        if args.export_only:
            export();return
        if not args.force and not should_train():
            print('No new reviewed dataset; keeping previous candidate',flush=True);return
        manifest=source_manifest()
        state={'dataset_id':manifest['id'],'status':'running','attempt_day':datetime.now().strftime('%Y%m%d'),
               'started':datetime.now().isoformat(),'pid':os.getpid()}
        atomic_json(STATE,state)
        try:
            out,manifest=export()
            n=manifest['counts']
            if n['train']['frames']<200 or not n['val']['frames']:
                state.update(status='insufficient_data',counts=n);atomic_json(STATE,state)
                print('Not enough complete reviewed frames:',n,flush=True);return
            from ultralytics import YOLO
            name='student_seg_'+manifest['id']
            print('Training candidate',name,n,flush=True)
            model=YOLO(args.model)
            model.train(data=str(out/'data.yaml'),imgsz=args.imgsz,epochs=args.epochs,batch=args.batch,
                        device=0,workers=8,patience=15,project=str(ROOT/'runs'),name=name,
                        exist_ok=True,pretrained=True,seed=0,val=True,plots=False)
            best=ROOT/'runs'/name/'weights/best.pt'
            if not best.exists():raise RuntimeError('Training produced no best checkpoint')
            res=YOLO(str(best)).val(data=str(out/'data.yaml'),imgsz=args.imgsz,device=0,plots=False)
            result={'when':datetime.now().isoformat(timespec='seconds'),'dataset_id':manifest['id'],
                    'frames_train':n['train']['frames'],'frames_val':n['val']['frames'],'imgsz':args.imgsz,
                    'weights':str(best),'mask_mAP50':float(res.seg.map50),'mask_mAP':float(res.seg.map),
                    'box_mAP50':float(res.box.map50),'box_mAP':float(res.box.map),
                    'status':'candidate','ground_truth':'teacher masks, not an independent pixel-level test'}
            atomic_json(ROOT/'data/student_seg_last.json',result)
            state.update(status='complete',result=result);atomic_json(STATE,state)
            print(json.dumps(result),flush=True)
        except Exception as exc:
            state.update(status='failed',error=str(exc)[-1000:]);atomic_json(STATE,state)
            raise


if __name__=='__main__':main()
