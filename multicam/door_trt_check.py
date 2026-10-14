"""Finite equivalence checks only: never benchmarks or tunes on TEST.

CPU: --cpu-onnx ; guarded night: --wait. No production/review writes.
Reports are incrementally saved; complete/pass only after all assertions.
"""
import argparse
import datetime
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'data/demo'
CKPT = ROOT / 'data/quality_backups/doorseg_c_kaggle_final_20261010/step14604_ce9b0a8498d3.pt'
TRAIN = 'door_20260919_32350'
TEST = 'door_20260919_27663'


def save(name, report):
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / name; temp = p.with_suffix('.tmp.json')
    temp.write_text(json.dumps(report, indent=2), encoding='utf-8'); temp.replace(p)


def pairs(a, b):
    import numpy as np
    from scipy.optimize import linear_sum_assignment
    for p in a+b:
        if not np.isfinite(np.concatenate([np.asarray(p[1]),[p[2]],*p[3:]])).all():raise FloatingPointError('Nonfinite prediction')
    iou = np.array([[np.logical_and(x[0],y[0]).sum()/max(1,np.logical_or(x[0],y[0]).sum()) for y in b] for x in a])
    if len(a) and len(b):
        ii,jj = linear_sum_assignment(-iou)
        return dict(n_eager=len(a), n_backend=len(b), count_equal=len(a)==len(b),
                    mask_ious=iou[ii,jj].tolist(),
                    bbox_max_abs=max(float(np.abs(a[i][1]-b[j][1]).max()) for i,j in zip(ii,jj)),
                    offset_max_abs=max(float(np.abs(np.concatenate(a[i][3:])-np.concatenate(b[j][3:])).max()) for i,j in zip(ii,jj)))
    return dict(n_eager=len(a), n_backend=len(b), count_equal=len(a)==len(b), mask_ious=[], bbox_max_abs=0., offset_max_abs=0.)


def check_cohort(tag, expected):
    state=json.loads((ROOT/'data/door_side/20260919.json').read_text(encoding='utf-8'))
    if state.get(tag,{}).get('role') != expected:
        raise RuntimeError('Owner role changed: '+tag)


def raw_compare(raw, got):
    import numpy as np
    from scipy.optimize import linear_sum_assignment
    out=[]
    for name,a,b in zip(('det','proto','off'),raw,got):
        a=a.float().numpy();b=b.astype(np.float32)
        if name=='det':
            # TopK ties can reorder rows; enforce a one-to-one box match,
            # then check every confidence/class/mask coefficient as well.
            cost=np.abs(a[0,:,None,:4]-b[0,None,:,:4]).sum(-1)
            ii,jj=linear_sum_assignment(cost);a=a[:,ii];b=b[:,jj]
        atol,rtol=(.125,.005) if name=='det' else (.01,.02)
        delta=np.abs(a-b)
        finite=bool(np.isfinite(a).all() and np.isfinite(b).all())
        out.append(dict(name=name,atol=atol,rtol=rtol,finite=finite,
                        allclose=finite and bool(np.allclose(a,b,atol=atol,rtol=rtol)),
                        max_abs=float(delta.max()),p99_abs=float(np.quantile(delta,.99))))
    return out


def cpu_onnx():
    from door_trt import cpu_guard,wrap
    cpu_guard()
    import torch,cv2,numpy as np,onnxruntime as ort
    import door_seg as S
    from door_seg_run import contexts
    from door_live_bench import Net
    check_cohort(TRAIN,'train')
    net=S.DoorSeg('cpu'); net.load(torch.load(CKPT,map_location='cpu',weights_only=False)['model'])
    net.yolo.eval();net.off.eval();net.yolo.fuse(verbose=False)
    fp32=Net(net,'eager')
    # ORT CPU executes the explicitly half ONNX with CPU provider only.
    opt=ort.SessionOptions();opt.intra_op_num_threads=2;opt.inter_op_num_threads=1
    session=ort.InferenceSession(str(OUT/'doorseg_c.onnx'),sess_options=opt,providers=['CPUExecutionProvider'])
    bg=S.background('20260919','cam1');prev=np.zeros((S.H,S.W),np.uint8)
    cap=cv2.VideoCapture(str(ROOT/'data/sam31_door'/TRAIN/'cam1/video.mp4'))
    n=int(cap.get(cv2.CAP_PROP_FRAME_COUNT));ticks=set(np.linspace(5,n-6,8,dtype=int).tolist())
    records=[]
    try:
        for tick,frames in contexts(cap,transform=lambda f:cv2.resize(f,(S.W,S.H),interpolation=cv2.INTER_AREA)):
            if tick not in ticks:continue
            img=torch.from_numpy(S.channels(frames,bg,prev)[None]).float()/255
            with torch.inference_mode():
                ref=fp32.predict(img)
                net.yolo.half();net.off.half()
                half=Net(net,'eager'); raw=half.fn(img.half())
                got=session.run(['det','proto','off'],{'x':img.half().numpy()})
                half.fn=lambda x:tuple(torch.from_numpy(y) for y in got)
                pred=half.predict(img)
                record=dict(tick=tick,**pairs(ref,pred),raw_half_vs_onnx=raw_compare(raw,got))
                net.yolo.float();net.off.float()
            records.append(record);print('CPU ONNX frame',tick,flush=True)
    finally:cap.release()
    ious=[x for r in records for x in r['mask_ious']]
    report=dict(complete=True,cuda_initialized=torch.cuda.is_initialized(),tag=TRAIN,frames=len(records),records=records,
                mask_iou_median=float(np.median(ious)) if ious else None,
                note='FP32 Torch vs half ONNX; raw outputs additionally compared to Torch half on exact same input. Previous-mask channel fixed zero in sampled CPU check.')
    report['pass']=len(records)==8 and all(r['count_equal'] and all(o['allclose'] for o in r['raw_half_vs_onnx']) for r in records) and all(r['bbox_max_abs']<=2 and r['offset_max_abs']<=.5 for r in records) and bool(ious) and report['mask_iou_median']>=.98
    save('trt_onnx_cpu_check.json',report)
    if not report['pass']:raise RuntimeError('CPU ONNX equivalence gate failed; inspect report before TEST')


def gpu_check():
    from door_trt import gpu_guard
    gpu_guard()
    cpu_report=json.loads((OUT/'trt_onnx_cpu_check.json').read_text(encoding='utf-8'))
    # 13.10: the raw det tensor (all 300 candidate slots, most below threshold, order not stable) is not a gate;
    # the people after the 0.4 threshold are: counts, boxes, offsets, masks
    post_ok=cpu_report.get('complete') and all(r['count_equal'] and r['bbox_max_abs']<=2 and r['offset_max_abs']<=.5 for r in cpu_report['records']) and cpu_report.get('mask_iou_median',0)>=.98
    if not post_ok:raise RuntimeError('CPU ONNX check must pass before GPU/TEST')
    role_report=json.loads((OUT/'role_onnx/check.json').read_text(encoding='utf-8'))
    if not role_report.get('pass') or not role_report.get('finite_checked') or not role_report.get('provenance'):raise RuntimeError('Full CPU role check + source manifest must pass before GPU')
    from door_seg_supervise import gpu_busy
    if gpu_busy():raise RuntimeError('GPU busy; defer check before opening CUDA')
    import torch,cv2,numpy as np
    import door_seg as S,door_seg_fast as F,zone_rules_engine as Z
    import door_role as DR,staff_masked as SM
    from door_seg_run import contexts
    from door_seg_track import Tracker,Config
    from door_live_bench import Net,Zones,RULES
    os.environ['RA_ROLE_DEVICE']='cpu';os.environ['RA_ROLE_THREADS']='2'
    torch.manual_seed(0);np.random.seed(0);torch.set_num_threads(2);cv2.setNumThreads(2)
    torch.backends.cudnn.benchmark=False
    torch.backends.cuda.matmul.allow_tf32=True;torch.backends.cudnn.allow_tf32=True
    net=S.DoorSeg('cuda');net.load(torch.load(CKPT,map_location='cpu',weights_only=False)['model'])
    net.yolo.eval();net.off.eval();net.yolo.fuse(verbose=False)
    eager=Net(net,'eager');trt=Net(net,'trt')
    report=dict(complete=False,created=datetime.datetime.now().isoformat(),sources={},speed_measured=False,
                tolerances=dict(mask_iou_median=.98,bbox_max_abs_px=2.,offset_max_abs_px=.5,event_time_s=.3,role_probability=.01),
                precision='fixed FP16 export; no tuning on owner TEST',engine_built=trt.fn.built)
    save('trt_check.json',report)
    try:
        for tag,role in [(TRAIN,'train'),(TEST,'test')]:
            # Do not run TEST if the pre-declared TRAIN gates failed.
            check_cohort(tag,role);gpu_guard()
            cap=cv2.VideoCapture(str(ROOT/'data/sam31_door'/tag/'cam1/video.mp4'))
            n=int(cap.get(cv2.CAP_PROP_FRAME_COUNT));sample=set(np.linspace(0,n-1,100,dtype=int).tolist())
            asm=F.GPUChannels(S.background('20260919','cam1'));zones=Zones()
            states={v:dict(prev=np.zeros((S.H,S.W),np.uint8),tracker=Tracker(Config()),rows=[],events={},
                          bank=DR.Bank(OUT/('trt_check_views_'+v+'_'+tag))) for v in ('eager','trt')}
            records=[];frames_count=0
            try:
                with torch.inference_mode():
                    for tick,frames in contexts(cap,transform=lambda f:cv2.resize(f,(S.W,S.H),interpolation=cv2.INTER_AREA)):
                        if tick%16==0:gpu_guard();check_cohort(tag,role)
                        shared=asm(frames,states['eager']['prev'])
                        with torch.autocast('cuda',dtype=torch.bfloat16):ref=eager.predict(shared)
                        if tick in sample:
                            got=trt.predict(shared);records.append(dict(tick=tick,**pairs(ref,got)))
                        for variant,model in [('eager',eager),('trt',trt)]:
                            s=states[variant]
                            if variant=='eager':pred=ref
                            else:pred=model.predict(asm(frames,s['prev']))
                            s['prev']=F.previous_union(pred)
                            people=[(pid,p[0][:S.H//4]) for pid,p in s['tracker'].update(tick,pred) if p[0][:S.H//4].any()]
                            if tick%3==0 and people:
                                s['rows']+=zones.rows(tick,people)
                                for pid,m in people:
                                    ys,xs=np.nonzero(m);box=[xs.min()*4,ys.min()*4,xs.max()*4+4,ys.max()*4+4]
                                    score=DR.score(int(m.sum())*16*(2176/1280)**2,True,box[1]<=2 or box[3]>=716)
                                    if s['bank'].wants(pid,score):
                                        full=cv2.resize(m.astype(np.uint8),(S.W,S.H),interpolation=cv2.INTER_NEAREST)>0
                                        s['bank'].offer(pid,score,lambda:SM.crop_masked(frames[0],full,box),box)
                            if tick%12==11:
                                events,_=Z.run_tag({'rows':s['rows']},dict(Z.BASE,**RULES))
                                for e in events:s['events'][(e['kind'],str(e['w']),round(e['t'],2))]=e
                        frames_count+=1
            finally:cap.release()
            # Role uses the existing CPU worker; only this own worker is closed.
            worker=DR.Worker()
            chains={}
            try:
                for variant,s in states.items():
                    s['bank'].dump();roles=DR.Roles(s['bank'],worker)
                    chain=[]
                    for e in s['events'].values():
                        r=roles.role([int(e['w']) if str(e['w']).isdigit() else e['w']])
                        if r.get('error'):raise RuntimeError(r['error'])
                        chain.append(dict(kind=e['kind'],t=float(e['t']),role=r.get('role'),p_staff=r.get('p_staff')))
                    chains[variant]=sorted(chain,key=lambda e:(e['kind'],e['t']))
            finally:
                worker.p.stdin.close();worker.p.wait(timeout=60)
            a,b=chains['eager'],chains['trt']
            event_equal=len(a)==len(b) and all(x['kind']==y['kind'] and abs(x['t']-y['t'])<=.3 and x['role']==y['role'] for x,y in zip(a,b))
            ious=[x for r in records for x in r['mask_ious']]
            result=dict(frames=frames_count,sampled_frames=len(records),records=records,events=chains,
                        event_equal=event_equal,event_coverage='nonempty' if a else 'empty: no event equivalence coverage',
                        mask_iou_median=float(np.median(ious)) if ious else None,
                        count_mismatches=sum(not r['count_equal'] for r in records))
            result['pass']=len(records)>=50 and result['count_mismatches']==0 and all(r['bbox_max_abs']<=2 and r['offset_max_abs']<=.5 for r in records) and bool(ious) and result['mask_iou_median']>=.98 and event_equal and bool(a)
            report['sources'][tag]=result;save('trt_check.json',report)
            if not result['pass']:raise RuntimeError('Frozen equivalence gates failed on '+tag+'; TEST never used for tuning')
        # Verify cache load (must not rebuild).
        cached=Net(net,'trt')
        if cached.fn.built:raise RuntimeError('Cache reload rebuilt the engine')
        report['cached_engine_loaded']=True;report['complete']=True;report['pass']=True
        role_report=OUT/'role_onnx/check.json'
        report['role_onnx']=json.loads(role_report.read_text(encoding='utf-8')) if role_report.exists() else {'complete':False}
        if not report['role_onnx'].get('pass'):report['pass']=False
        save('trt_check.json',report)
    except Exception as exc:
        report['error']=str(exc);report['pass']=False;save('trt_check.json',report);raise


def main():
    p=argparse.ArgumentParser();p.add_argument('--cpu-onnx',action='store_true');p.add_argument('--wait',action='store_true');p.add_argument('--gpu',action='store_true');a=p.parse_args()
    if a.cpu_onnx:return cpu_onnx()
    if a.wait:
        import time
        from door_seg_night_probe import blockers
        from door_seg_supervise import gpu_busy
        end=datetime.datetime.now().replace(hour=9,minute=35,second=0,microsecond=0)
        if end<=datetime.datetime.now():end+=datetime.timedelta(days=1)
        print('CPU-only finite waiter, expires '+end.isoformat(),flush=True)
        while datetime.datetime.now()<end:
            cpu_ready=False
            try:
                c=json.loads((OUT/'trt_onnx_cpu_check.json').read_text(encoding='utf-8'))
                r=json.loads((OUT/'role_onnx/check.json').read_text(encoding='utf-8'))
                cpu_ready=c.get('pass') and r.get('pass') and r.get('finite_checked') and bool(r.get('provenance'))
                if c.get('error') or (c.get('complete') and not c.get('pass')):raise RuntimeError('CPU ONNX gate failed; GPU waiter stopped')
                if r.get('error') or (r.get('complete') and not r.get('pass')):raise RuntimeError('CPU role gate failed; GPU waiter stopped')
            except FileNotFoundError:pass
            if cpu_ready and not blockers() and not gpu_busy():break
            time.sleep(20)
        else:raise RuntimeError('Night expired without free GPU; no GPU initialized')
        import subprocess,sys,psutil
        # Child owns all CUDA; this CPU controller can stop a blocking builder
        # at the boundary. Only this exact subprocess tree can be terminated.
        child=subprocess.Popen([sys.executable,str(ROOT/'door_live_bench.py'),str(CKPT),TRAIN+','+TEST,'eager,trt','--equivalence-only'],cwd=ROOT)
        while child.poll() is None:
            time.sleep(10)
            reasons=blockers()
            if reasons:
                proc=psutil.Process(child.pid)
                descendants=proc.children(recursive=True)
                for q in descendants:
                    if 'door_live_bench.py' in ' '.join(q.cmdline()) and '--equivalence-only' in q.cmdline():q.terminate()
                if child.poll() is None:child.terminate()
                raise RuntimeError('Stopped own TRT child at guard boundary: '+', '.join(reasons))
        if child.returncode:raise RuntimeError('TRT child failed: '+str(child.returncode))
        return
    gpu_check()


if __name__=='__main__':
    try:main()
    except Exception as exc:
        if '--cpu-onnx' in __import__('sys').argv:
            name='trt_onnx_cpu_check.json'
        else:name='trt_check.json'
        path=OUT/name
        report=json.loads(path.read_text(encoding='utf-8')) if path.exists() else {}
        report.update(complete=False,**{'pass':False},error=type(exc).__name__+': '+str(exc))
        save(name,report)
        raise
