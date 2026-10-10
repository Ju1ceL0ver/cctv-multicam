"""One full separate temporal trainer, frozen148 TRAIN/6 VAL, yielding nightly.

CPU-only controller before GPU guards pass. BF16 AMP ordered four-frame unroll,
all parameters trained, restored Adam history, finite rollback, atomic backup,
fixed400 validation every2000 successful updates, three stale evaluations.
C STOP/checkpoints and production are never modified.
"""
import argparse
import collections
import datetime
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time

import door_seg_night_probe as P

ROOT=Path(__file__).resolve().parent
SOURCE=ROOT/'data/quality_backups/doorseg_c_kaggle_final_20261010/last_085f995430fb.pt'
SOURCE_SHA='085f995430fbe3a973c9a9f460460be9e61f41cda6b02bc432abb4f03ac24ae8'


class YieldGPU(RuntimeError): pass


def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()


def atomic_json(p,data):
    tmp=p.with_suffix('.tmp.json');tmp.write_text(json.dumps(data,indent=2),encoding='utf-8');tmp.replace(p)


_busy_checked=(0.,[])


def busy():
    global _busy_checked
    now=time.monotonic()
    if now-_busy_checked[0]<2:return list(_busy_checked[1])
    found=P.blockers()
    import psutil
    own={os.getpid(),*(p.pid for p in psutil.Process().parents())}
    for p in psutil.process_iter(['pid','name']):
        if p.info['pid'] in own:continue
        if 'python' not in (p.info['name'] or '').lower():continue
        try:cmd=' '.join(p.cmdline())
        except psutil.NoSuchProcess:continue
        if any(x in cmd for x in ('door_seg_sequence_train.py','door_seg_sequence_probe.py','auto_night.py','train_student_seg.py','staff_learn.py','staff_train.py')):
            found.append('other GPU job PID %d'%p.info['pid'])
    _busy_checked=(now,found)
    return list(found)


def log(run,event,**data):
    row={'time':datetime.datetime.now().isoformat(),'event':event,**data}
    with (run/'progress.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(row)+'\n')
    print(json.dumps(row),flush=True)


def stage(run):
    source=json.loads((ROOT/'doorseg_sequence_frozen_cohort.json').read_text(encoding='utf-8'))
    if len(source['train'])!=148 or len(source['val'])!=6 or set(source['train'])&set(source['val']):
        raise ValueError('frozen148 TRAIN/6 VAL required')
    if any('20260923' in p for p in source['train']):raise ValueError('validation leaked into TRAIN')
    files=[row for row in source['files'] if row['path'].startswith('data/')]
    frozen=[]
    for row in files:
        p=ROOT/row['path'];st=p.stat()
        if st.st_size!=row['size']:raise ValueError('frozen data size changed: '+row['path'])
        item={'path':row['path'],'bytes':st.st_size,'mtime_ns':st.st_mtime_ns}
        if p.suffix in ('.json','.jpg'):item['sha256']=sha(p)
        frozen.append(item)
    result={'version':1,'train':source['train'],'val':source['val'],'files':frozen,'source_sha256':SOURCE_SHA,
            'architecture':'fingerprint64/ConvGRU48','precision':'bf16','matmul_precision':'high','sequence_length':4,'batch_sequences':1,
            'identity_weight':.1,'base_lr':1e-4,'head_lr':1e-4,'all_parameters_trained':True,
            'validation_moments':400,'validation_interval':2000,'plateau_stale':3,'min_delta':.005}
    atomic_json(run/'train_manifest.json',result)
    return result


def worker(run):
    manifest=json.loads((run/'train_manifest.json').read_text(encoding='utf-8'))
    manifest_sha=sha(run/'train_manifest.json')
    pending=busy()
    if pending and all(x.startswith('sampling foreign app') for x in pending):
        time.sleep(3);pending=busy()
    if pending:return 2
    def guard():
        if not P.night(datetime.datetime.now()) or (run/'PAUSE').exists():raise YieldGPU('day boundary/controller pause')
        blocked=busy()
        if blocked:raise YieldGPU(', '.join(blocked))
    guard()
    import torch,cv2
    import door_seg as S
    from door_seg_sequence import DoorSegTemporal,sequence_starts,sample_sequence,sequence_loss
    from door_seg_sequence_state import restore,save,finite
    from door_seg_sequence_update import update
    from door_seg_sequence_eval import evaluate
    torch.set_num_threads(2);cv2.setNumThreads(2);torch.manual_seed(7)
    torch.set_float32_matmul_precision('high')
    torch.backends.cuda.matmul.allow_tf32=True
    torch.backends.cudnn.allow_tf32=True
    torch.backends.cudnn.benchmark=True
    os.environ['RA_DS_GLUED']='1'
    train=[ROOT/p for p in manifest['train']];val=[ROOT/p for p in manifest['val']]
    eligible={p.resolve() for p in S.sources()}
    if any(p.resolve() not in eligible for p in train):raise ValueError('frozen TRAIN source no longer eligible')
    current=json.loads((run/'state.json').read_text()) if (run/'state.json').exists() else {'successful_updates':0,'stale':0,'best_f1':None}
    checkpoint=run/'last.pt' if (run/'last.pt').exists() else SOURCE
    if checkpoint==SOURCE and sha(SOURCE)!=SOURCE_SHA:raise ValueError('source SHA differs')
    saved=torch.load(checkpoint,map_location='cpu',weights_only=False);finite(saved)
    if saved['precision'] not in ('fp32','bf16'):raise ValueError('saved precision missing')
    net=DoorSegTemporal('cuda',weights=str(ROOT/'yolo26s-seg.pt'));opt,resume=restore(net,saved)
    recipe={**manifest,'train_manifest_sha256':manifest_sha,'loss_precision':'fp32'}
    if 'temporal' in saved['model'] and saved['sequence_recipe']['train_manifest_sha256']!=manifest_sha:
        raise ValueError('resume frozen manifest differs')
    step=saved['step'];del saved
    net.yolo.train();net.off.train();net.temporal.train()
    criterion=net.yolo.init_criterion()
    for _ in range(step//2000):criterion.update()
    excluded=set(current.get('no_ordered_sources', []))
    active_train=[p for p in train if str(p.relative_to(ROOT)) not in excluded]
    if not active_train:raise ValueError('no ordered TRAIN sources remain')
    rng=random.Random(7+current['successful_updates']);cache=collections.OrderedDict()
    data_checked=0.
    def data_guard():
        nonlocal data_checked
        guard()
        if time.monotonic()-data_checked>30:
            for item in manifest['files']:
                p=ROOT/item['path'];st=p.stat()
                if st.st_size!=item['bytes'] or st.st_mtime_ns!=item['mtime_ns']:
                    raise ValueError('frozen data changed: '+item['path'])
            data_checked=time.monotonic()
    def persist():
        save(net,opt,run/'last.pt',step=step,precision='bf16',recipe=recipe)
        atomic_json(run/'state.json',current)
    def validation():
        log(run,'validation_start',step=step,successful_updates=current['successful_updates'],moments=400)
        result=evaluate(net,val,n=400,guard=data_guard,progress=lambda row:log(run,'validation_progress',step=step,**row))
        atomic_json(run/('validation_%d.json'%step),result)
        summary={k:v for k,v in result.items() if k!='per_moment'}
        log(run,'validation_complete',step=step,successful_updates=current['successful_updates'],**summary)
        if current['best_f1'] is None or result['f1']>=current['best_f1']+.005:
            current.update(best_f1=result['f1'],best_step=step,stale=0)
            save(net,opt,run/'best.pt',step=step,precision='bf16',recipe={**recipe,'validation':summary})
        else:current['stale']+=1
        current['last_validation_updates']=current['successful_updates'];persist()
        net.yolo.train();net.off.train();net.temporal.train()
    try:
        persist();log(run,'trainer_ready',pid=os.getpid(),source_step=resume['step'],train_sources=148,val_sources=6,architecture=net.temporal.arch,precision='bf16',amp=True,matmul_precision='high',loss_precision='fp32',all_parameters_trained=True)
        if 'last_validation_updates' not in current:validation()
        saved_at=time.monotonic()
        while current['stale']<3:
            data_guard();chosen=rng.choice(active_train)
            key=str(chosen)
            if key not in cache:
                src=S.Source(chosen);starts=sequence_starts(src,4)
                if not starts:
                    src.close();active_train.remove(chosen)
                    current.setdefault('no_ordered_sources', []).append(str(chosen.relative_to(ROOT)))
                    log(run,'source_without_ordered_window',source=str(chosen.relative_to(ROOT)),remaining_sources=len(active_train))
                    persist()
                    if not active_train:raise ValueError('no ordered TRAIN sources remain')
                    continue
                cache[key]=(src,starts)
                if len(cache)>8:
                    _,(old,_)=cache.popitem(last=False);old.close()
            src,starts=cache.pop(key);cache[key]=(src,starts)
            at=rng.choice(starts);seed=rng.randrange(2**31)
            frames=sample_sequence(src,4,seed=seed,at=at,allowed_sources=train)
            def loss_fn():
                with torch.autocast('cuda',dtype=torch.bfloat16):
                    return sequence_loss(net,[frames],criterion,'cuda',before_frame=data_guard)
            try:
                result=update(net,opt,loss_fn,guard=data_guard)
            except YieldGPU:raise
            except Exception as exc:
                torch.save({'sequences':[frames],'step':step,'reason':str(exc)},run/('rejected_%d.pt'%step))
                persist();(run/'STOP').write_text('numerical/technical rejection: '+str(exc),encoding='utf-8');raise
            step+=1;current['successful_updates']+=1
            if step%2000==0:criterion.update()
            if current['successful_updates']<=3 or current['successful_updates']%20==0:
                log(run,'train_update',step=step,successful_updates=current['successful_updates'],source=str(chosen.relative_to(ROOT)),tick=at[1],**result,peak_gpu_bytes=torch.cuda.max_memory_allocated())
            if time.monotonic()-saved_at>=120:
                persist();saved_at=time.monotonic();log(run,'backup',step=step,successful_updates=current['successful_updates'])
            if current['successful_updates']-current['last_validation_updates']>=2000:validation()
        (run/'STOP').write_text('plateau: three spaced400 validations without +.005',encoding='utf-8');persist();return 0
    except YieldGPU as exc:
        persist();log(run,'yield_gpu',step=step,reason=str(exc));return 2
    except Exception as exc:
        # Preserve the last accepted update even when sampling/validation fails.
        persist();log(run,'technical_exit',step=step,reason=str(exc));raise
    finally:
        for src,_ in cache.values():src.close()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--worker',action='store_true');parser.add_argument('--check',action='store_true')
    parser.add_argument('--run',default='doorseg_sequence_full_20261011');args=parser.parse_args()
    if not args.run.startswith('doorseg_sequence_'):raise ValueError('separate sequence run required')
    run=ROOT/'runs'/args.run
    if args.worker:return worker(run)
    if args.check:
        print(json.dumps({'blockers':busy(),'torch_imported':'torch' in sys.modules,'source_exists':SOURCE.exists()}));return 0
    run.mkdir(exist_ok=True)
    if (run/'STOP').exists():raise RuntimeError('STOP requires investigation; never clear automatically')
    import psutil
    lock=run/'controller.pid'
    if lock.exists():
        try:
            p=psutil.Process(int(lock.read_text()))
            if 'door_seg_sequence_train.py' in ' '.join(p.cmdline()):raise RuntimeError('controller already running')
        except psutil.NoSuchProcess:pass
    lock.write_text(str(os.getpid()))
    try:
        if not (run/'train_manifest.json').exists():stage(run)
        end=datetime.datetime.now().replace(hour=9,minute=35,second=0,microsecond=0)
        if end<=datetime.datetime.now():end+=datetime.timedelta(days=1)
        from door_seg_supervise import gpu_busy
        while datetime.datetime.now()<end and not (run/'STOP').exists():
            if busy() or gpu_busy():time.sleep(10);continue
            (run/'PAUSE').unlink(missing_ok=True)
            with (run/'worker.log').open('a',encoding='utf-8') as output:
                cmd=[sys.executable,str(Path(__file__).resolve()),'--worker','--run',args.run]
                p=subprocess.Popen(cmd,cwd=ROOT,stdout=output,stderr=subprocess.STDOUT)
                log(run,'worker_started',pid=p.pid)
                while p.poll() is None:
                    if datetime.datetime.now()>=end:
                        (run/'PAUSE').write_text('day boundary',encoding='utf-8')
                        try:p.wait(timeout=45)
                        except subprocess.TimeoutExpired:
                            child=psutil.Process(p.pid)
                            if str(Path(__file__).resolve()) not in child.cmdline() or '--worker' not in child.cmdline():raise RuntimeError('child identity changed')
                            subprocess.run(['taskkill','/PID',str(p.pid),'/T','/F'],check=True,capture_output=True);p.wait(timeout=10)
                        break
                    time.sleep(5)
                log(run,'worker_exit',code=p.returncode)
                if p.returncode not in (0,2):return p.returncode
            time.sleep(10)
        return 0
    finally:lock.unlink(missing_ok=True)


if __name__=='__main__':sys.exit(main())
