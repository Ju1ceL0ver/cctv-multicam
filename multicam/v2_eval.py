"""The v2 model against the owner's 18.09 painted frames (the exam, never trained on) -- the same scoring as
train_slots.exam for v1, so the numbers compare. The painted frames are 1280 x 720: the detail branch gets
them enlarged (a handicap v1 did not have: its whole input was 1088 x 608).

usage: v2_eval.py CHECKPOINT [--limit N]"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))


def empty_like(r):
    import torch
    return {k: (v[:, :0] if torch.is_tensor(v) and v.dim() >= 2 else v) for k, v in r.items()}


def predict(model, asm, img, bg, cam, dev):
    """One frame (BGR, any size) -> [(score, mask bool 1280 x 720)]."""
    import torch
    import torch.nn.functional as F
    import v2_data as VD
    f = {'img': cv2.resize(img, (VD.FW, VD.FH), interpolation=cv2.INTER_LINEAR),
         'bg_long': bg, 'bg_now': cv2.resize(bg, (VD.BG_W if hasattr(VD, 'BG_W') else 1088, 612)) if bg is not None else None}
    if f['bg_now'] is None:
        f['bg_now'] = cv2.resize(img, (1088, 612))
    with torch.no_grad(), torch.autocast('cuda', torch.bfloat16, enabled=dev == 'cuda'):
        rgb, bgv, sta, cam_ids = asm([f], (cam,), False)
        bg_sem = model.body.vit(bgv)[0]
        m = model.maps(rgb, sta, cam_ids, bg_sem, asm.last_world)
        r = model.decode(m, cam_ids)
        r, _ = model.cross_cameras(r, empty_like(r))
        masks = model.full_masks(m, r)[0].float()
        p = r['obj'][0].float().sigmoid()
    masks = masks[:, :VD.FH // 4]
    out = []
    for s in range(len(p)):
        full = F.interpolate(masks[s][None, None], size=(720, 1280), mode='bilinear', align_corners=False)[0, 0] > 0
        out.append((float(p[s]), full.cpu().numpy()))
    return out


def exam(model, dev, limit=None, thr=0.3):
    import gold
    import train_slots as TS
    import v2_data as VD
    split = json.load(open(ROOT / 'data' / 'logs' / 'gold_all.json'))['split']['test_ids']
    where = json.load(open(ROOT / 'data' / 'seg_datasets' / 'backgrounds' / 'exam_frames.json'))
    ids = split[:limit] if limit else split
    asm = VD.Assembler(dev, getattr(model, 'rgb_size', None))
    was = model.training
    model.eval()
    found = total = false = small_found = small_total = any_found = found_d = false_d = 0
    ious = []
    for ident in ids:
        w = where.get(ident, {})
        cam = w.get('cam', 'cam1')
        full = ROOT / 'data' / 'v2_exam' / ('%s.jpg' % ident)        # the raw frame at the training resolution
        img = cv2.imread(str(full if full.exists() else gold.PAINT / ('%s.jpg' % ident)))
        bgp = ROOT / 'data' / 'seg_datasets' / 'backgrounds' / ('%s_%s_%s.jpg' % (w.get('day'), cam, str(w.get('segment', ''))[:-4]))
        bg = cv2.imread(str(bgp)) if bgp.exists() else None
        people = predict(model, asm, img, bg, cam, dev)
        truth = gold.gold(ident)
        pred = [m for sc, m in people if m.sum() >= gold.MIN_PX]
        score = [sc for sc, m in people if m.sum() >= gold.MIN_PX]
        pairs_any, _ = gold.match(truth, pred)
        any_found += len(pairs_any)
        keep = [j for j, sc in enumerate(score) if sc >= thr]
        kd = [keep[i] for i in TS.dedup([pred[j] for j in keep], [score[j] for j in keep])]
        pd_, _ = gold.match(truth, [pred[j] for j in kd])
        found_d += len(pd_); false_d += len(kd) - len(pd_)
        pairs, iou = gold.match(truth, [pred[j] for j in keep])
        total += len(truth); found += len(pairs); false += len(keep) - len(pairs)
        ious += [iou[i, j] for i, j in pairs]
        hit = {i for i, _ in pairs}
        for i, t in enumerate(truth):
            ys = np.nonzero(t.any(1))[0]
            if ys[-1] - ys[0] < gold.SMALL:
                small_total += 1; small_found += i in hit
    model.train(was)
    return {'frames': len(ids), 'people': total, 'thr': thr, 'recall': round(found / max(1, total), 4), 'false': false,
            'precision': round(found / max(1, found + false), 4),
            'mask_iou_median': round(float(np.median(ious)), 4) if ious else None,
            'small_recall': round(small_found / max(1, small_total), 4), 'recall_any': round(any_found / max(1, total), 4),
            'dedup': {'recall': round(found_d / max(1, total), 4), 'false': false_d, 'precision': round(found_d / max(1, found_d + false_d), 4)}}


def tracking(model, dev, tag, start=0, ticks=1500, delay_s=0.0):
    """The runtime tracker on a held-out window against SAM's people (per camera, a window's person = one
    identity): IDF1 (best one-to-one mapping of world ids to people, by mask IoU >= 0.5 per frame), identity
    switches (a person's matched world id changes), found share of people-frames."""
    from scipy.optimize import linear_sum_assignment
    import v2_data as VD
    import v2_track as VT
    rt = VT.Runtime(model, dev, delay_s=delay_s)
    w = [VD.Window(tag, c) for c in ('cam1', 'cam2')]
    counts = {}
    gt_n, pr_n, found = 0, 0, 0
    last = {}
    switches = 0
    for t in range(start, start + ticks):
        frames = []
        for x in w:
            img = x.frame(t)
            if img is None:
                break
            frames.append({'img': img, 'bg_long': x.bg_long(t), 'bg_now': x.bg_now(t)})
        if len(frames) < 2:
            continue
        ans, _ = rt.step(t, frames, masks_out=True)
        for c in range(2):
            tg = w[c].targets(t)
            mine = [p for p in ans['people'] if p['cam'] == c]
            gt_n += len(tg['person']); pr_n += len(mine)
            if not len(mine) or not len(tg['person']):
                continue
            G = tg['masks']
            P = np.stack([p['mask'][:G.shape[1], :G.shape[2]] for p in mine])
            inter = np.einsum('ghw,phw->gp', G.astype(np.float32), P.astype(np.float32))
            union = G.reshape(len(G), -1).sum(1)[:, None] + P.reshape(len(P), -1).sum(1)[None] - inter
            iou = inter / np.maximum(union, 1)
            r, cc = linear_sum_assignment(-iou)
            for i, j in zip(r, cc):
                if iou[i, j] < 0.5:
                    continue
                found += 1
                key = (c, int(tg['person'][i]))
                wid = mine[j]['world']
                counts[(key, wid)] = counts.get((key, wid), 0) + 1
                if key in last and last[key] != wid:
                    switches += 1
                last[key] = wid
    gts = sorted({k for k, _ in counts}); pds = sorted({v for _, v in counts})
    M = np.zeros((len(gts), len(pds)))
    for (k, v), n in counts.items():
        M[gts.index(k), pds.index(v)] = n
    r, c = linear_sum_assignment(-M) if M.size else ([], [])
    idtp = float(M[r, c].sum()) if M.size else 0.0
    return {'tag': tag, 'ticks': ticks, 'people_frames': gt_n, 'found': round(found / max(1, gt_n), 4),
            'idf1': round(2 * idtp / max(1, gt_n + pr_n), 4), 'switches': switches, 'people': len({k[1] for k in gts}),
            'world_ids': len(pds)}


if __name__ == '__main__':
    import torch
    import slot_v2 as V
    ck = torch.load(ROOT / sys.argv[1], map_location='cpu')
    m = V.build()
    m.load_state_dict(ck.get('ema', ck['model']))
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    lim = int(sys.argv[sys.argv.index('--limit') + 1]) if '--limit' in sys.argv else None
    if '--track' in sys.argv:
        tag = sys.argv[sys.argv.index('--track') + 1]
        print(json.dumps(tracking(m.to(dev), dev, tag)))
    else:
        print(json.dumps(exam(m.to(dev), dev, lim)))
