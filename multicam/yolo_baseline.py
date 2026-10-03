"""The yardstick run (30.09): an ordinary detector, yolo26s-seg, trained on the SAM 3.1 windows (yolo_sam_export.py)
and scored after every epoch on the same teacher test as the slot model (v2_teacher_test's 160 moments of 23.09, the
stride-4 grid, IoU >= 0.5 one to one, duplicates dropped). Written like a train_v2 run -- runs/<out>/status.json,
log.jsonl, epochs.jsonl with 'test' -- so the /train page and the variant queue see it; the queue's STOP file ends it
after the epoch.

usage: yolo_baseline.py --out runs/NAME [--model PT] [--epochs N] [--imgsz S] [--batch B]"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))


def teacher_score(yolo, test, imgsz, conf=0.25):
    """v2_teacher_test's score for a YOLO model (its masks at the frame's size -> the stride-4 grid)."""
    import train_slots as TS
    import v2_data as VD
    from scipy.optimize import linear_sum_assignment
    rec = {'all': [0, 0], 'hall': [0, 0], 'out': [0, 0], 'small': [0, 0], 'busy': [0, 0]}
    false, ious = 0, []
    for x in test.items:
        img = cv2.imdecode(x['jpg'], cv2.IMREAD_COLOR)
        r = yolo.predict(img, imgsz=imgsz, conf=conf, classes=[0], retina_masks=True, verbose=False, half=True)[0]
        P = []
        if r.masks is not None:
            for m in r.masks.data.cpu().numpy():
                g = cv2.resize(m.astype(np.float32), (VD.GW, VD.FH // 4), interpolation=cv2.INTER_AREA) >= 0.5
                full = np.zeros((VD.GH, VD.GW), bool)
                full[:VD.FH // 4] = g
                if full.sum() >= 20:
                    P.append(full.reshape(-1))
        n = len(x['zone'])
        G = np.unpackbits(x['masks'], 1)[:, :VD.GH * VD.GW].astype(bool) if n else np.zeros((0, VD.GH * VD.GW), bool)
        P = [P[j] for j in TS.dedup(P, [1.0] * len(P))] if P else []
        hit = {}
        if n and P:
            Gf, Pf = G.astype(np.float32), np.stack(P).astype(np.float32)
            inter = Gf @ Pf.T
            iou = inter / np.maximum(Gf.sum(1)[:, None] + Pf.sum(1)[None] - inter, 1)
            rr, cc = linear_sum_assignment(-iou)
            hit = {i: j for i, j in zip(rr, cc) if iou[i, j] >= 0.5}
            ious += [float(iou[i, j]) for i, j in hit.items()]
        false += len(P) - len(hit)
        for i in range(n):
            parts = ['all'] + (['busy'] if x['busy'] else []) + {1: ['hall'], 0: ['out']}.get(int(x['zone'][i]), []) + (['small'] if x['small'][i] else [])
            for k in parts:
                rec[k][0] += int(i in hit)
                rec[k][1] += 1
    f, t = rec['all']
    R, Pr = f / max(1, t), f / max(1, f + false)
    out = {'frames': len(test.items), 'people': t, 'recall': round(R, 4), 'precision': round(Pr, 4),
           'f1': round(2 * R * Pr / max(1e-9, R + Pr), 4), 'false': false,
           'mask_iou_median': round(float(np.median(ious)), 4) if ious else None}
    for k in ('hall', 'out', 'small', 'busy'):
        out['recall_' + k] = round(rec[k][0] / max(1, rec[k][1]), 4)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='')
    ap.add_argument('--model', default='runs/student_seg_all_s/weights/best.pt')
    ap.add_argument('--epochs', type=int, default=10)
    ap.add_argument('--imgsz', type=int, default=1088)
    ap.add_argument('--batch', type=int, default=6)
    ap.add_argument('--test', default='v2b')
    ap.add_argument('--data', default='data/seg_datasets/sam31_yolo/data.yaml')
    ap.add_argument('--lr0', type=float, default=0.0, help='0: Ultralytics default; a fine-tune wants a small one')
    ap.add_argument('--warmup', type=float, default=-1.0)
    ap.add_argument('--extra', nargs='*', default=[], help='more Ultralytics train settings, KEY=VALUE (freeze=10 mosaic=0)')
    ap.add_argument('--score', default='', help='only score these weights on the teacher test, print the JSON (a separate process)')
    a, _ = ap.parse_known_args()                   # the queue's common arguments are the slot model's: ignored
    if a.score:
        import v2_teacher_test
        from ultralytics import YOLO
        print('SCORE ' + json.dumps(teacher_score(YOLO(a.score), v2_teacher_test.Test(a.test), a.imgsz)), flush=True)
        return
    import torch
    from ultralytics import YOLO
    out = ROOT / a.out
    out.mkdir(parents=True, exist_ok=True)
    json.dump(dict(vars(a), kind='yolo'), open(out / 'args.json', 'w'), indent=1)
    t0 = time.time()
    state = {'epoch_t0': time.time(), 'best': -1.0}
    data = ROOT / a.data
    model = YOLO(str(ROOT / a.model) if (ROOT / a.model).exists() else 'yolo26s-seg.pt')

    def status(extra):
        st = {'step': 0, 'steps': state.get('total', a.epochs), 'epoch': 0, 'epoch_steps': state.get('per', 1), 'elapsed_s': round(time.time() - t0), 'eta_s': 0,
              'until': '', 'updated': time.strftime('%H:%M:%S'), 'phase': 'training'}
        st.update(extra)
        json.dump(st, open(out / 'status.json', 'w'), indent=1)

    def gstep(trainer):
        return trainer.epoch * max(1, len(trainer.train_loader)) + state.get('batch', 0)

    def on_batch(trainer):
        """A log row every 20 batches in the train_v2 form (step = batches so far), so the /train page draws it like
        the slot model's runs: speed, rate, memory, losses."""
        state['batch'] = state.get('batch', 0) + 1
        g = gstep(trainer)
        now = time.time()
        if g % 20:
            return
        dt = (now - state.get('row_t', now)) / 20
        state['row_t'] = now
        loss = {k.split('/')[-1]: round(float(v), 4) for k, v in trainer.label_loss_items(trainer.tloss, prefix='train').items()}
        with open(out / 'log.jsonl', 'a') as f:
            f.write(json.dumps(dict(step=g, t=round(now - t0), frozen=False, kind='yolo', lr=float(trainer.optimizer.param_groups[0]['lr']),
                                    data_s=0.0, gpu_s=round(dt, 3), mem_gb=round(torch.cuda.max_memory_allocated() / 2 ** 30, 2), **loss)) + '\n')
        total = a.epochs * max(1, len(trainer.train_loader))
        state['total'], state['per'] = total, max(1, len(trainer.train_loader))
        status({'step': g, 'steps': total, 'epoch': trainer.epoch + 1, 'epoch_steps': max(1, len(trainer.train_loader)),
                'eta_s': round((now - t0) / max(1, g) * (total - g)), 'step_s': round(dt, 3), 'phase': 'training'})

    def on_epoch_start(trainer):
        state['batch'] = 0

    def on_epoch(trainer):
        ep = trainer.epoch + 1
        loss = {k.split('/')[-1]: round(float(v), 4) for k, v in trainer.label_loss_items(trainer.tloss, prefix='train').items()}
        g = ep * max(1, len(trainer.train_loader))
        status({'step': g, 'epoch': ep, 'phase': 'testing epoch %d' % ep})
        w = Path(trainer.last)
        rec = {'epoch': ep, 'step': g, 'minutes': round((time.time() - state['epoch_t0']) / 60, 1), 'frozen': False, 'loss': loss,
               'time': time.strftime('%H:%M')}
        try:                                           # in a process of its own: scoring inside the training process
            import shutil                              # turned some weights into doubles (30.09: the EMA update died)
            import subprocess
            snap = out / 'scoring.pt'
            shutil.copy2(w, snap)
            r = subprocess.run([sys.executable, __file__, '--score', str(snap), '--imgsz', str(a.imgsz), '--test', a.test],
                               capture_output=True, text=True, cwd=str(ROOT))
            line = [l for l in r.stdout.splitlines() if l.startswith('SCORE ')]
            if not line:
                raise RuntimeError((r.stdout + r.stderr)[-300:])
            rec['test'] = json.loads(line[-1][6:])
            if rec['test']['f1'] > state['best']:
                state['best'] = rec['test']['f1']
                rec['best'] = True
                shutil.copy2(snap, out / 'best.pt')
        except Exception as e:
            rec['test'] = {'error': repr(e)[:300]}
        with open(out / 'epochs.jsonl', 'a') as f:
            f.write(json.dumps(rec) + '\n')
        state['epoch_t0'] = time.time()
        status({'step': g, 'epoch': ep, 'phase': 'training'})
        print('EPOCH', json.dumps(rec), flush=True)
        if (out / 'STOP').exists():                        # the variant queue: end after this epoch
            (out / 'STOP').unlink()
            trainer.stop = True
            status({'step': g, 'epoch': ep, 'phase': 'stopped'})

    def to_float(trainer):
        """30.09: from our own checkpoint, after epoch 1 some tensors of the model come back as float64 and the EMA
        update (torch._foreach_lerp_) dies at the first step of epoch 2 -- twice, with and without our scoring."""
        import torch
        for m in (trainer.model, getattr(trainer.ema, 'ema', None)):
            if m is None:
                continue
            for t in list(m.parameters()) + list(m.buffers()):
                if t.dtype == torch.float64:
                    t.data = t.data.float()

    model.add_callback('on_train_batch_end', on_batch)
    model.add_callback('on_train_epoch_start', on_epoch_start)
    model.add_callback('on_fit_epoch_end', on_epoch)
    model.add_callback('on_train_epoch_start', to_float)
    model.add_callback('on_fit_epoch_end', to_float)
    status({'phase': 'training'})
    model.train(data=str(data), imgsz=a.imgsz, epochs=a.epochs, batch=a.batch, workers=4, device=0, project=str(out),
                name='yolo', exist_ok=True, plots=False, val=True, patience=100, cos_lr=True,
                **({'lr0': a.lr0} if a.lr0 else {}), **({'warmup_epochs': a.warmup} if a.warmup >= 0 else {}),
                **{k: (int(v) if v.isdigit() else float(v) if v.replace('.', '', 1).isdigit() else v) for k, v in (e.split('=', 1) for e in a.extra)})
    status({'step': a.epochs, 'epoch': a.epochs, 'phase': 'done'})


if __name__ == '__main__':
    main()
