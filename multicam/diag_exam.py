"""Why is the exam score what it is? Error breakdown of the current model on the owner's 18.09 frames.

  python diag_exam.py runs/v2_a/epoch_30.pt

For every person of the exam: found at threshold 0.3 / a slot covers them but scores low / no slot covers them (by size, camera,
crowding). For every predicted person over the threshold that matches nobody: a duplicate of a found person / a partial overlap
with one / nothing there at all (by size). And the precision-recall curve over the threshold (after removing duplicates)."""
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import gold                      # noqa: E402
import slot_v2 as V              # noqa: E402
import train_slots as TS         # noqa: E402
import v2_data as VD             # noqa: E402
import v2_eval as VE             # noqa: E402

ck = sys.argv[1] if len(sys.argv) > 1 else 'runs/v2_a/epoch_30.pt'
dev = 'cuda'
model = V.build()
model.load_state_dict(torch.load(ROOT / ck, map_location='cpu')['ema'])
model.to(dev).eval().to(memory_format=torch.channels_last)
asm = VD.Assembler(dev)
split = json.load(open(ROOT / 'data' / 'logs' / 'gold_all.json'))['split']['test_ids']
where = json.load(open(ROOT / 'data' / 'seg_datasets' / 'backgrounds' / 'exam_frames.json'))
THR = 0.3
sizes = [(0, 60, '<60 px'), (60, 120, '60-120'), (120, 240, '120-240'), (240, 10 ** 9, '>240')]
sz = lambda h: next(n for a, b, n in sizes if a <= h < b)
height = lambda m: (lambda ys: int(ys[-1] - ys[0]) if len(ys) else 0)(np.nonzero(m.any(1))[0])
gt_rows, fp_rows = [], []
curve = defaultdict(lambda: [0, 0, 0])            # thr -> [found, predicted, truth]
for ident in split:
    w = where.get(ident, {})
    cam = w.get('cam', 'cam1')
    full = ROOT / 'data' / 'v2_exam' / ('%s.jpg' % ident)
    img = cv2.imread(str(full if full.exists() else gold.PAINT / ('%s.jpg' % ident)))
    bgp = ROOT / 'data' / 'seg_datasets' / 'backgrounds' / ('%s_%s_%s.jpg' % (w.get('day'), cam, str(w.get('segment', ''))[:-4]))
    bg = cv2.imread(str(bgp)) if bgp.exists() else None
    people = VE.predict(model, asm, img, bg, cam, dev)
    truth = gold.gold(ident)
    pred = [m for sc, m in people if m.sum() >= gold.MIN_PX]
    score = [sc for sc, m in people if m.sum() >= gold.MIN_PX]
    if not pred:
        for t in truth:
            gt_rows.append((cam, sz(height(t)), 'no slot', 0.0, 0, len(truth)))
        continue
    # IoU of every truth with every prediction
    inter = lambda a, b: float((a & b).sum()) / max(1, float((a | b).sum()))
    M = np.array([[inter(t, p) for p in pred] for t in truth]) if truth else np.zeros((0, len(pred)))
    keep = [j for j, s in enumerate(score) if s >= THR]
    kd = [keep[i] for i in TS.dedup([pred[j] for j in keep], [score[j] for j in keep])]
    pairs, _ = gold.match(truth, [pred[j] for j in kd])
    hit = {i for i, _ in pairs}
    for i, t in enumerate(truth):
        best = int(M[i].argmax()) if M.shape[1] else -1
        bi = M[i, best] if best >= 0 else 0
        if i in hit:
            cat = 'found'
        elif bi >= 0.5:
            cat = 'a slot covers them, score %.2f < %.1f' % (score[best], THR) if score[best] < THR else 'covered but lost by duplicate removal'
        elif bi >= 0.2:
            cat = 'a slot overlaps them badly (IoU 0.2-0.5)'
        else:
            cat = 'no slot near them'
        near = int((np.array([[inter(t, o) for o in truth]]) > 0.05).sum()) - 1 if len(truth) > 1 else 0     # neighbours touching
        gt_rows.append((cam, sz(height(t)), cat, float(bi), near, len(truth)))
    matched_pred = {kd[j] for _, j in pairs}
    for j in kd:
        if j in matched_pred:
            continue
        best = int(M[:, j].argmax()) if len(truth) else -1
        bi = M[best, j] if best >= 0 else 0
        cat = 'duplicate of a found person' if best in hit and bi >= 0.4 else 'partial overlap with a person' if bi >= 0.15 else 'nothing there'
        fp_rows.append((cam, sz(height(pred[j])), cat, score[j]))
    for thr in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9):
        kp = [j for j, s in enumerate(score) if s >= thr]
        kp = [kp[i] for i in TS.dedup([pred[j] for j in kp], [score[j] for j in kp])] if kp else []
        pr, _ = gold.match(truth, [pred[j] for j in kp])
        curve[thr][0] += len(pr); curve[thr][1] += len(kp); curve[thr][2] += len(truth)

n = len(gt_rows)
print('people in the exam: %d   (checkpoint %s, threshold %.1f, duplicates removed)\n' % (n, ck, THR))
c = Counter(r[2] for r in gt_rows)
print('every person of the exam:')
for k, v in c.most_common():
    print('  %-46s %4d  %4.1f %%' % (k, v, 100 * v / n))
print('\nby size (height in the 720 p frame): found / all')
for _, _, name in sizes:
    a = [r for r in gt_rows if r[1] == name]
    print('  %-8s %3d / %3d = %4.1f %%   never covered by any slot: %d' % (name, sum(r[2] == 'found' for r in a), len(a), 100 * sum(r[2] == 'found' for r in a) / max(1, len(a)), sum(r[2] == 'no slot near them' for r in a)))
print('\nby camera:')
for cam in ('cam1', 'cam2'):
    a = [r for r in gt_rows if r[0] == cam]
    print('  %s %3d people, found %4.1f %%' % (cam, len(a), 100 * sum(r[2] == 'found' for r in a) / max(1, len(a))))
print('\nby crowding (people touching this one):')
for lo, hi, name in ((0, 0, 'alone'), (1, 1, '1 neighbour'), (2, 99, '2+ neighbours')):
    a = [r for r in gt_rows if lo <= r[4] <= hi]
    print('  %-14s %3d people, found %4.1f %%' % (name, len(a), 100 * sum(r[2] == 'found' for r in a) / max(1, len(a))))
print('\nfalse people over the threshold (%d):' % len(fp_rows))
for k, v in Counter(r[2] for r in fp_rows).most_common():
    print('  %-34s %4d   sizes: %s' % (k, v, dict(Counter(r[1] for r in fp_rows if r[2] == k))))
print('\nprecision-recall over the threshold (duplicates removed):')
for thr in sorted(curve):
    f, p, t = curve[thr]
    print('  thr %.1f   found %5.1f %%   precision %5.1f %%   (%d found of %d, %d predicted)' % (thr, 100 * f / max(1, t), 100 * f / max(1, p), f, t, p))
json.dump({'gt': gt_rows, 'fp': fp_rows, 'curve': {str(k): v for k, v in curve.items()}}, open(ROOT / 'data' / 'logs' / 'diag_exam.json', 'w'))
