"""The v2 model's other heads on what it never trained on (03.10.2026). One cold frame each, as v2_teacher_test.

reid    the owner's door pairs 17-19.09 (door_review/<day>_clean.json): for each exit, is its own entry the most
        alike among everyone inside then (eval_slot_heads.test_reid's rule, so v1 and the teachers compare). The
        counter's full snapshot goes through the model; the person is the slot whose box fits the event's best.
        Vectors: the whole identity, and its two halves when it is 256 + 256 (clothes, body shape).
zone    the owner's /inout answers on 23.09 (the held-out day): inside (1) / outside (2, and 0 'cannot tell': people
        in other shops, as inout_train) / doorway (3); reflections and 'not a person' left out
place   floor place and zone against the teacher's targets (v2_prep: feet on the floor, door line) on the 160
        moments of the teacher test (23.09), people with the feet seen

usage: v2_heads_eval.py CKPT  -> data/logs/v2_heads_<run>.json"""
import datetime
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
THR = 0.3


def cold(model, asm, dev, img, cam, bg_long, bg_now):
    """-> list of slots: (score, mask stride-4 bool GH x GW, box cx cy w h of the padded frame, ident, zone probs, xy)."""
    import torch
    import slot_v2 as V
    import v2_data as VD
    import v2_eval
    f = {'img': img, 'bg_long': bg_long, 'bg_now': bg_now}
    with torch.no_grad(), torch.autocast('cuda', torch.bfloat16, enabled=dev == 'cuda'):
        rgb, bgv, sta, cam_ids = asm([f], (cam,), False)
        bg_sem = model.body.vit(bgv)[0]
        m = model.maps(rgb, sta, cam_ids, bg_sem, asm.last_world)
        r = model.decode(m, cam_ids)
        r, _ = model.cross_cameras(r, v2_eval.empty_like(r))
        logits = model.full_masks(m, r)[0]
        pool = model.pooled(m, r, logits.float().sigmoid()[None].to(logits.dtype))
    p = r['obj'][0].float().sigmoid().cpu().numpy()
    pad = r['pad'][0].cpu().numpy()
    masks = (logits > 0).cpu().numpy()
    box = r['box'][0].float().cpu().numpy()
    ident = pool['ident'][0].float().cpu().numpy()
    zone = pool['zone'][0].float().softmax(-1).cpu().numpy()
    place = r['place'][0].float().cpu().numpy()
    xy = place[:, :2] * V.XY_S + np.array(V.XY_C)
    import train_slots as TS
    keep = [s for s in range(len(p)) if p[s] >= THR and not pad[s] and masks[s].sum() >= 20]
    keep = [keep[i] for i in TS.dedup([masks[s] for s in keep], [float(p[s]) for s in keep])]
    return [(float(p[s]), masks[s], box[s], ident[s], zone[s], xy[s]) for s in keep]


def miou(a, b):
    u = np.logical_or(a, b).sum()
    return np.logical_and(a, b).sum() / u if u else 0.0


def bgs_of(day, cam):
    out = []
    for p in (ROOT / 'data' / 'seg_datasets' / 'backgrounds').glob('%s_%s_*.jpg' % (day, cam)):
        hms, seg = p.stem.split('_')[2:4]
        out.append((int(hms[:2]) * 3600 + int(hms[2:4]) * 60 + int(hms[4:]) + 900 * int(seg) + 450, p))
    return sorted(out)


def test_reid(model, asm, dev, days=('20260917', '20260918', '20260919')):
    import door_review
    import v2_data as VD
    res = {}
    for day in days:
        clean = json.load(open(ROOT / 'data' / 'door_review' / ('%s_clean.json' % day), encoding='utf-8'))['visits']
        events = {e['event_id']: e for e in door_review.day_events(day, str(ROOT))}
        ids = sorted({v['entry'] for v in clean} | {v['exit'] for v in clean})
        bgs = bgs_of(day, 'cam1')
        vec, miss = {}, 0
        for i in ids:
            e = events[i]
            lt = datetime.datetime.fromisoformat(e['time_local'])
            sec = lt.hour * 3600 + lt.minute * 60 + lt.second
            bg = cv2.imread(str(min(bgs, key=lambda b: abs(b[0] - sec))[1]))
            img = cv2.imread(e['snapshot_full'])
            if img is None or bg is None:
                miss += 1; vec[i] = None; continue
            slots = cold(model, asm, dev, cv2.resize(img, (VD.FW, VD.FH), interpolation=cv2.INTER_AREA), 'cam1', bg, cv2.resize(bg, (1088, 612)))
            k = VD.FW / img.shape[1]
            eb = [e['box_x1'] * k, e['box_y1'] * k, e['box_x2'] * k, e['box_y2'] * k]
            best, bi = None, 0.0
            for s in slots:
                cx, cy, w, h = s[2]
                pb = [(cx - w / 2) * VD.FW, (cy - h / 2) * VD.PH, (cx + w / 2) * VD.FW, (cy + h / 2) * VD.PH]
                ix = max(0.0, min(eb[2], pb[2]) - max(eb[0], pb[0])); iy = max(0.0, min(eb[3], pb[3]) - max(eb[1], pb[1]))
                v = ix * iy / max(1e-9, (eb[2] - eb[0]) * (eb[3] - eb[1]) + (pb[2] - pb[0]) * (pb[3] - pb[1]) - ix * iy)
                if v > bi:
                    best, bi = s, v
            if best is None or bi < 0.3:
                miss += 1; vec[i] = None; continue
            vec[i] = best[3]
        t = {i: events[i]['unix_ms'] / 1000 for i in ids}
        dim = next((len(v) for v in vec.values() if v is not None), 0)
        parts = {'whole': slice(0, dim)}
        if dim == 512:
            parts.update(clothes=slice(0, 256), shape=slice(256, 512))
        day_res = {'exits': len(clean), 'person_not_found': miss, 'dim': dim}
        for name, sl in parts.items():
            nv = lambda x: x[sl] / max(1e-9, np.linalg.norm(x[sl]))
            right = hard = hard_right = scored = 0
            for v in clean:
                if vec[v['exit']] is None or vec[v['entry']] is None:
                    continue
                inside = [w['entry'] for w in clean if t[w['entry']] < t[v['exit']] <= t[w['exit']] + 0.5 and vec[w['entry']] is not None]
                ex = nv(vec[v['exit']])
                d = {e_: 1 - float(ex @ nv(vec[e_])) for e_ in inside}
                best_e = min(d, key=d.get)
                scored += 1; right += best_e == v['entry']
                if len(inside) > 1:
                    hard += 1; hard_right += best_e == v['entry']
            day_res[name] = {'scored': scored, 'right_first': right, 'crowded': hard, 'crowded_right_first': hard_right}
        res[day] = day_res
    names = [k for k in res[days[0]] if isinstance(res[days[0]][k], dict)]
    res['all'] = {n: {k: sum(res[d][n][k] for d in days) for k in ('scored', 'right_first', 'crowded', 'crowded_right_first')} for n in names}
    return res


def test_zone(model, asm, dev, day='20260923'):
    import inout
    import rate
    import v2_data as VD
    S, L = inout.samples(str(ROOT)), inout.labels(str(ROOT))
    frames = rate.index(str(ROOT), force=True)
    want = {1: 1, 2: 0, 0: 0, 3: 2}                  # owner's key -> the model's zone (0 out, 1 in, 2 doorway)
    by = {}
    for sid, lab in L.items():
        s = S.get(sid)
        k = lab.get('label') if isinstance(lab, dict) else lab
        if s is None or s['day'] != day or k not in want or s['frame'] not in frames:
            continue
        by.setdefault(s['frame'], []).append((s['value'], want[k]))
    conf = np.zeros((3, 3), int)
    miss = 0
    for ident, ppl in by.items():
        img_p, lab_p = frames[ident]
        img = cv2.imread(str(img_p)); lab = cv2.imread(str(lab_p), cv2.IMREAD_UNCHANGED)
        if lab.ndim == 3:
            lab = lab[:, :, 0]
        parts = ident.split('_')
        cam = parts[1]
        bgp = ROOT / 'data' / 'seg_datasets' / 'backgrounds' / ('%s_%s_%s.jpg' % (parts[0], cam, '_'.join(parts[2:4])))
        bg = cv2.imread(str(bgp))
        if bg is None:
            bs = bgs_of(parts[0], cam)
            bg = cv2.imread(str(bs[len(bs) // 2][1])) if bs else None
        if img is None or bg is None:
            miss += len(ppl); continue
        slots = cold(model, asm, dev, img, cam, bg, cv2.resize(bg, (1088, 612)))
        for v, truth in ppl:
            g = cv2.resize((lab == v).astype(np.float32), (VD.GW, VD.FH // 4), interpolation=cv2.INTER_AREA) >= 0.5
            tm = np.zeros((VD.GH, VD.GW), bool); tm[:VD.FH // 4] = g
            best, bi = None, 0.0
            for s in slots:
                u = miou(s[1], tm)
                if u > bi:
                    best, bi = s, u
            if best is None or bi < 0.3:
                miss += 1; continue
            conf[truth, int(np.argmax(best[4]))] += 1
    n = conf.sum()
    names = ['outside', 'inside', 'doorway']
    return {'day': day, 'people': int(n), 'not_found': miss, 'accuracy3': round(float(np.trace(conf) / max(1, n)), 4),
            'recall': {names[i]: round(float(conf[i, i] / max(1, conf[i].sum())), 3) for i in range(3)},
            'confusion_rows_truth': conf.tolist(), 'inside_vs_outside_wrong': int(conf[0, 1] + conf[1, 0])}


def test_place(model, asm, dev):
    import v2_data as VD
    from scipy.optimize import linear_sum_assignment
    spec = json.load(open(ROOT / 'data' / 'v2_test' / 'v2b.json'))['items']
    wins, err, zc = {}, {'feet': [], 'head': []}, np.zeros((3, 3), int)
    for it in spec:
        k = (it['tag'], it['cam'])
        if k not in wins:
            wins[k] = VD.Window(*k)
        w = wins[k]
        img = w.frame(it['tick'])
        if img is None:
            continue
        tg = w.targets(it['tick'])
        slots = cold(model, asm, dev, img, it['cam'], w.bg_long(it['tick']), np.array(w.bg_now(it['tick'])))
        n = len(tg['masks'])
        if not n or not slots:
            continue
        iou = np.array([[miou(tg['masks'][i], s[1]) for s in slots] for i in range(n)])
        r, c = linear_sum_assignment(-iou)
        for i, j in zip(r, c):
            if iou[i, j] < 0.5:
                continue
            if tg['zone'][i] >= 0:
                zc[int(tg['zone'][i]), int(np.argmax(slots[j][4]))] += 1
            if np.isfinite(tg['xy'][i]).all() and tg['place_w'][i] > 0:
                d = float(np.linalg.norm(slots[j][5] - tg['xy'][i]))
                err['feet' if tg['place_w'][i] >= 1.0 else 'head'].append(d)
    for w in wins.values():
        if w.cap is not None:
            w.cap.release()
    q = lambda a: {'n': len(a), 'median_m': round(float(np.median(a)), 3), 'p90_m': round(float(np.percentile(a, 90)), 3)} if a else None
    return {'place_vs_teacher': {k: q(v) for k, v in err.items()},
            'zone_vs_teacher': {'people': int(zc.sum()), 'accuracy3': round(float(np.trace(zc) / max(1, zc.sum())), 4),
                                'confusion_rows_truth': zc.tolist()}}


def main():
    import torch
    import slot_v2 as V
    import v2_data as VD
    ck = sys.argv[1]
    dev = 'cuda'
    model = V.load(str(ROOT / ck)).to(dev).eval()
    asm = VD.Assembler(dev, getattr(model, 'rgb_size', None))
    rep = {'checkpoint': ck, 'started': time.strftime('%H:%M:%S')}
    out = ROOT / 'data' / 'logs' / ('v2_heads_%s_%s.json' % (Path(ck).parent.name, Path(ck).stem))
    save = lambda: json.dump(rep, open(out, 'w'), indent=1)
    for name, fn in (('zone_owner', test_zone), ('place', test_place), ('reid', test_reid)):
        t0 = time.time()
        try:
            rep[name] = fn(model, asm, dev)
        except Exception as e:                            # one head failing must not lose the others
            import traceback
            rep[name] = {'error': traceback.format_exc()[-1500:]}
        rep[name]['s'] = round(time.time() - t0)
        save()
    rep['finished'] = time.strftime('%H:%M:%S'); save()


if __name__ == '__main__':
    main()
