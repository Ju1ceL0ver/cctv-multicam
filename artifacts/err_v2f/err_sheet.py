"""Every miss and every false of a model on the teacher test, as numbered tiles to judge by eye.
SAM's mask green, the model's red. CPU only. usage: err_sheet.py CKPT OUTNAME"""
import json
import os
import sys
from pathlib import Path

os.environ['CUDA_VISIBLE_DEVICES'] = ''
ROOT = Path('C:/Users/ArykovAA/cctv_ai/multicam')
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))
import cv2
import numpy as np
import torch

torch.set_num_threads(8)
import slot_v2 as V
import train_slots as TS
import v2_data as VD
import v2_eval
import v2_teacher_test as TT
from scipy.optimize import linear_sum_assignment

THR = 0.3
OUT = ROOT / 'data' / 'logs' / sys.argv[2]
OUT.mkdir(parents=True, exist_ok=True)
model = V.load(ROOT / sys.argv[1])
model.eval()
asm = VD.Assembler('cpu', getattr(model, 'rgb_size', None))
spec = json.load(open(TT.OUT / 'v2b.json'))['items']
test = TT.Test('v2b')
assert len(spec) == len(test.items), (len(spec), len(test.items))
print('test read', test.people, flush=True)
TW, TH = 380, 420


def tile(img, g, q, text):
    """g, q: masks on the stride-4 grid (either may be None)."""
    H, W = img.shape[:2]
    im = img.copy()
    box = None
    for mm, col in ((g, (0, 255, 0)), (q, (0, 0, 255))):
        if mm is None or not mm.any():
            continue
        up = cv2.resize(mm.astype(np.uint8), (mm.shape[1] * 4, mm.shape[0] * 4), interpolation=cv2.INTER_NEAREST)[:H, :W]
        cs, _ = cv2.findContours(up, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(im, cs, -1, col, 2)
        ys, xs = np.nonzero(up)
        if len(ys) and (box is None or mm is g or g is None):
            b = [xs.min(), ys.min(), xs.max(), ys.max()]
            box = b if box is None else [min(box[0], b[0]), min(box[1], b[1]), max(box[2], b[2]), max(box[3], b[3])]
    x0, y0, x1, y1 = box
    side = max(260, int(max(x1 - x0, y1 - y0) * 1.9))
    cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
    ax, ay = max(0, min(W - side, cx - side // 2)), max(0, min(H - side, cy - side // 2))
    crop = im[ay:ay + side, ax:ax + side]
    s = TW / max(crop.shape[:2])
    crop = cv2.resize(crop, (int(crop.shape[1] * s), int(crop.shape[0] * s)), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC)
    t = np.zeros((TH, TW, 3), np.uint8)
    t[:crop.shape[0], :crop.shape[1]] = crop
    cv2.putText(t, text, (3, TH - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return t


miss, false, tot = [], [], {'people': 0, 'found': 0, 'kept': 0}
for k, (it, x) in enumerate(zip(spec, test.items)):
    img = cv2.imdecode(x['jpg'], cv2.IMREAD_COLOR)
    f = {'img': img, 'bg_long': x['bg_long'], 'bg_now': x['bg_now']}
    with torch.no_grad():
        rgb, bgv, sta, cam_ids = asm([f], (x['cam'],), False)
        bg_sem = model.body.vit(bgv)[0]
        m = model.maps(rgb, sta, cam_ids, bg_sem, asm.last_world)
        r = model.decode(m, cam_ids)
        r, _ = model.cross_cameras(r, v2_eval.empty_like(r))
        P = (model.full_masks(m, r)[0] > 0).numpy()
        p = r['obj'][0].float().sigmoid().numpy()
        pad = r['pad'][0].numpy()
    sh = x['shape']
    n = len(x['zone'])
    G = np.unpackbits(x['masks'], 1)[:, :sh[0] * sh[1]].astype(bool) if n else np.zeros((0, sh[0] * sh[1]), bool)
    P = P[:, :sh[0], :sh[1]].reshape(len(P), -1)
    ok = [s for s in range(len(p)) if not pad[s] and P[s].sum() >= TT.MIN_CELLS]
    P, p = P[ok], p[ok]
    iou = np.zeros((n, len(P)), np.float32)
    if n and len(P):
        Gf, Pf = G.astype(np.float32), P.astype(np.float32)
        inter = Gf @ Pf.T
        iou = inter / np.maximum(Gf.sum(1)[:, None] + Pf.sum(1)[None] - inter, 1)
    keep = [j for j in range(len(p)) if p[j] >= THR]
    keep = [keep[i] for i in TS.dedup([P[j] for j in keep], [p[j] for j in keep])]
    hit = {}
    if n and keep:
        rr, cc = linear_sum_assignment(-iou[:, keep])
        hit = {int(i): keep[j] for i, j in zip(rr, cc) if iou[i, keep[j]] >= 0.5}
    tot['people'] += n; tot['found'] += len(hit); tot['kept'] += len(keep)
    zn = lambda i: {1: 'hall', 0: 'out', 2: 'door'}.get(int(x['zone'][i]), '?')
    for i in range(n):
        if i in hit:
            continue
        j = int(np.argmax(iou[i])) if len(P) else -1
        best = float(iou[i, j]) if j >= 0 else 0.0
        sc = float(p[j]) if j >= 0 else 0.0
        bk = max([float(iou[i, j2]) for j2 in keep], default=0.0)
        cat = ('lowconf' if sc < THR else 'taken') if best >= 0.5 else 'badmask' if best >= 0.25 else 'unseen'
        d = {'item': k, 'tag': it['tag'], 'cam': x['cam'], 'tick': it['tick'], 'zone': zn(i), 'small': bool(x['small'][i]), 'busy': bool(x['busy']),
             'scrap': bool(x['scrap'][i]), 'cells': int(G[i].sum()), 'best_iou': round(best, 3), 'best_score': round(sc, 3), 'best_kept_iou': round(bk, 3), 'cat': cat}
        miss.append((d, img, G[i].reshape(sh), P[j].reshape(sh) if j >= 0 and best > 0.05 else None))
    used = set(hit.values())
    for j in keep:
        if j in used:
            continue
        gi = int(np.argmax(iou[:, j])) if n else -1
        best = float(iou[gi, j]) if gi >= 0 else 0.0
        d = {'item': k, 'tag': it['tag'], 'cam': x['cam'], 'tick': it['tick'], 'score': round(float(p[j]), 3), 'cells': int(P[j].sum()), 'best_iou': round(best, 3),
             'on_found': bool(gi in hit) if gi >= 0 else False, 'on_scrap': bool(x['scrap'][gi]) if gi >= 0 and best >= 0.25 else False,
             'cat': 'on_teacher_person' if best >= 0.25 else 'touch' if best >= 0.05 else 'empty'}
        ys, xs = np.nonzero(P[j].reshape(sh))
        d['box'] = [int(xs.min() * 4), int(ys.min() * 4), int(xs.max() * 4), int(ys.max() * 4)]
        false.append((d, img, G[gi].reshape(sh) if gi >= 0 and best >= 0.05 else None, P[j].reshape(sh)))
    if k % 20 == 0:
        print(k, len(miss), len(false), flush=True)

order = {'hall': 0, 'door': 1, 'out': 2, '?': 3}
miss.sort(key=lambda t: (order[t[0]['zone']], t[0]['cat'], -t[0]['best_iou']))
false.sort(key=lambda t: (t[0]['cat'], -t[0]['score']))
for name, L in (('miss', miss), ('false', false)):
    tiles = []
    for q, (d, img, g, pm) in enumerate(L):
        d['n'] = q
        if name == 'miss':
            txt = '#%d c%s %s %s i%.2f s%.2f%s' % (q, d['cam'][-1], d['zone'], d['cat'], d['best_iou'], d['best_score'], ' SCRAP' if d['scrap'] else '')
        else:
            txt = '#%d c%s %s i%.2f s%.2f' % (q, d['cam'][-1], d['cat'], d['best_iou'], d['score'])
        tiles.append(tile(img, g, pm, txt))
    for s in range(0, len(tiles), 15):
        part = tiles[s:s + 15] + [np.zeros((TH, TW, 3), np.uint8)] * (15 - len(tiles[s:s + 15]))
        cv2.imwrite(str(OUT / ('%s_%02d.jpg' % (name, s // 15))), np.vstack([np.hstack(part[a:a + 5]) for a in (0, 5, 10)]), [cv2.IMWRITE_JPEG_QUALITY, 88])
json.dump({'totals': tot, 'miss': [t[0] for t in miss], 'false': [t[0] for t in false]}, open(OUT / 'items.json', 'w'), indent=0)
print('done', tot, len(miss), len(false), flush=True)
