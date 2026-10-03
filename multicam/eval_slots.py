"""Score a training run of the slot model on the 401 held-out drafts, every time it saves an epoch.

The 401 frames come from 25 whole 15-minute recording files none of whose frames was trained on
(pseudo_20260925_test400 = rf_20260925/valid minus the 100 random first-pass frames, 92 of whose files
are also in training). Truth = the teacher+SAM draft, not the owner: this is the bigger, noisier check
next to the owner's 18.09 exam. Same numbers as the exam: people found at mask IoU >= 0.5, false, median
mask IoU, small people; person-ness >= thr.

usage: eval_slots.py RUN [--thr 0.3] [--once]   -> RUN/heldout.jsonl (one line per epoch)
Runs beside the training (a few minutes of the card per epoch) and stops by day or when the training
is gone and its last epoch is scored."""
import argparse
import datetime
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))


def held_out_ids(root=ROOT / 'data'):
    seg = Path(root) / 'seg_datasets'
    valid = {p.stem for p in (seg / 'rf_20260925' / 'valid' / 'images').glob('*.jpg')}
    t400 = {p.stem for p in (seg / 'pseudo_20260925_test400' / 'images').glob('*.jpg')}
    return sorted(valid & t400)


def score(model, frames, dev, ids, thr=0.3):
    import cv2
    import torch
    import torch.nn.functional as F
    import gold
    import slot_data
    from train_slots import predict, dedup
    found = total = false = small_found = small_total = any_found = found_d = false_d = 0
    ious = []
    model.eval()
    for ident in ids:
        s = slot_data.load(frames, ident, False)
        x = torch.from_numpy(s['x'])[None].to(dev)
        with torch.no_grad(), torch.autocast(dev.split(':')[0], torch.bfloat16, enabled=dev.startswith('cuda')):
            people = predict(model, x, thr=0.0)[0]
        L = cv2.imread(str(frames.drafts[ident]), cv2.IMREAD_UNCHANGED)
        L = L[:, :, 0] if L.ndim == 3 else L
        if L.shape != (slot_data.SH, slot_data.SW):
            L = cv2.resize(L, (slot_data.SW, slot_data.SH), interpolation=cv2.INTER_NEAREST)
        truth = [L == v for v in np.unique(L) if v and (L == v).sum() >= gold.MIN_PX]
        pred, sc_ = [], []
        for sc, m, *_ in people:
            full = F.interpolate(m[None, None], size=(slot_data.H, slot_data.W), mode='bilinear', align_corners=False)[0, 0].cpu().numpy()
            full = cv2.resize(full, (slot_data.SW, slot_data.SH), interpolation=cv2.INTER_LINEAR) > 0
            if full.sum() >= gold.MIN_PX:
                pred.append(full); sc_.append(sc)
        any_found += len(gold.match(truth, pred)[0])
        keep = [pred[j] for j, sc in enumerate(sc_) if sc >= thr]
        ks = [sc for sc in sc_ if sc >= thr]
        kd = [keep[i] for i in dedup(keep, ks)]
        pd_, _ = gold.match(truth, kd)
        found_d += len(pd_); false_d += len(kd) - len(pd_)
        pairs, iou = gold.match(truth, keep)
        total += len(truth); found += len(pairs); false += len(keep) - len(pairs)
        ious += [iou[i, j] for i, j in pairs]
        hit = {i for i, _ in pairs}
        for i, t in enumerate(truth):
            ys = np.nonzero(t.any(1))[0]
            if ys[-1] - ys[0] < gold.SMALL:
                small_total += 1; small_found += i in hit
    return {'frames': len(ids), 'people': total, 'thr': thr, 'recall': round(found / max(1, total), 4), 'false': false,
            'precision': round(found / max(1, found + false), 4),
            'mask_iou_median': round(float(np.median(ious)), 4) if ious else None,
            'small_recall': round(small_found / max(1, small_total), 4), 'recall_any': round(any_found / max(1, total), 4),
            'dedup': {'recall': round(found_d / max(1, total), 4), 'false': false_d, 'precision': round(found_d / max(1, found_d + false_d), 4)}}


def training_alive(run):
    import subprocess
    out = subprocess.run(['powershell', '-c', 'Get-CimInstance Win32_Process -Filter "name=\'python.exe\'" | %{ $_.CommandLine }'],
                         capture_output=True, text=True).stdout if os.name == 'nt' else ''
    return any('train_slots.py' in l and str(run).replace('\\', '/') in l.replace('\\', '/') for l in out.splitlines())


def main():
    import torch
    import slot_data
    from slot_model import SlotModel
    ap = argparse.ArgumentParser()
    ap.add_argument('run')
    ap.add_argument('--thr', type=float, default=0.3)
    ap.add_argument('--once', action='store_true')
    a = ap.parse_args()
    run = Path(a.run)
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    frames = slot_data.Frames()
    ids = held_out_ids()
    while not (run / 'args.json').exists():            # the training writes it once it has started
        time.sleep(20)
    args = json.load(open(run / 'args.json'))
    done = set()
    if (run / 'heldout.jsonl').exists():
        done = {json.loads(l)['epoch'] for l in open(run / 'heldout.jsonl')}
    seen = None
    while True:
        last = run / 'last.pt'
        stamp = last.stat().st_mtime if last.exists() else None
        if stamp and stamp != seen:
            seen = stamp
            ck = torch.load(last, map_location='cpu')
            if ck['epoch'] not in done:
                model = SlotModel(args['backbone'], pretrained=False, mean=frames.mean, std=frames.std, teacher_dims=frames.teacher_dims)
                model.load_state_dict(ck.get('ema', ck['model']))
                model = model.to(dev)
                t0 = time.time()
                res = {'epoch': ck['epoch'], 'at': time.strftime('%H:%M'), **score(model, frames, dev, ids, a.thr), 's': round(time.time() - t0)}
                with open(run / 'heldout.jsonl', 'a') as f:
                    f.write(json.dumps(res) + '\n')
                done.add(ck['epoch'])
                del model
                torch.cuda.empty_cache()
        if a.once:
            return
        t = datetime.datetime.now().time()
        if datetime.time(9, 45) <= t < datetime.time(21, 0):
            return
        if not training_alive(run) and (last.stat().st_mtime if last.exists() else None) == seen:
            return
        time.sleep(60)


if __name__ == '__main__':
    main()
