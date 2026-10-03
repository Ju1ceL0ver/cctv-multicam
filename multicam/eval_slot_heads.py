"""The slot model's other heads, measured on what it never trained on (27.09).

radio   cosine of the model's copy of C-RADIOv4-H (PCA-512) to the real one, per valid cell, on the 401
        held-out drafts; the same on 200 training frames for comparison
reid    the owner's door pairs of 17-19.09 (door_review/<day>_clean.json): for each exit, is its own entry
        the most alike among everyone inside at that moment (as reid_eval_shop.door_pairs, so the teachers'
        numbers compare). The whole 2560x1440 snapshot of the counter's event goes through the model; the
        person is the slot whose box fits the event's box best. Vectors: clothes, body shape, and their mixes.
        18.09 is the honest day (no draft of 18.09 was trained on).
zone    inside / outside / doorway on the owner's /inout answers of frames not trained on (the 401 held-out
        drafts and the drafts dropped for a low rating); the person = the slot whose mask fits their draft mask.

usage: eval_slot_heads.py CHECKPOINT  -> data/logs/eval_slot_heads.json"""
import datetime
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
MIXES = {'clothes': 1.0, 'shape': 0.0, 'mix_0.25': 0.25, 'mix_0.5': 0.5, 'mix_0.75': 0.75}   # weight of clothes


def run(model, s, dev, radio=False):
    import torch
    x = torch.from_numpy(s['x'])[None].to(dev)
    with torch.no_grad(), torch.autocast('cuda', torch.bfloat16, enabled=dev == 'cuda'):
        out = model(x, aux=False, radio=radio)
    return out


def people(model, out, thr=0.3):
    """Kept, de-duplicated people: (score, mask logits stride 4, box cx cy w h, clothes, shape, zone probs)."""
    import torch
    from train_slots import dedup
    with torch.no_grad():
        p = out['obj'][0].float().sigmoid()
        keep = torch.nonzero(p >= thr).flatten().tolist()
        if not keep:
            return []
        m = torch.einsum('sc,chw->shw', out['mask_vec'][0, keep], out['pix'][0])
        w = torch.zeros(1, p.shape[0], *m.shape[-2:], device=m.device, dtype=out['pix'].dtype)
        w[0, keep] = m.sigmoid().to(w.dtype)
        pooled = model.pooled(out, w)
    masks = (m > 0).cpu().numpy()
    kd = [keep[i] for i in dedup(list(masks), [float(p[k]) for k in keep])]
    res = []
    for k in kd:
        j = keep.index(k)
        res.append((float(p[k]), masks[j], out['box'][0, k].float().cpu().numpy(), pooled['cloth'][0, k].float().cpu().numpy(),
                    pooled['shape'][0, k].float().cpu().numpy(), pooled['inout'][0, k].float().softmax(-1).cpu().numpy()))
    return res


def box_iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    return inter / max(1e-9, (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter)


def test_radio(model, frames, dev, ids):
    import torch
    import torch.nn.functional as F
    import slot_data
    cos = []
    for ident in ids:
        s = slot_data.load(frames, ident, False)
        if not s['radio_valid'].any():
            continue
        out = run(model, s, dev, radio=True)
        c = F.cosine_similarity(out['radio'][0].float(), torch.from_numpy(s['radio']).float().to(dev), dim=0)
        cos.append(float(c[torch.from_numpy(s['radio_valid']).to(dev)].mean()))
    return {'frames': len(cos), 'cosine_mean': round(float(np.mean(cos)), 4) if cos else None,
            'cosine_p10': round(float(np.percentile(cos, 10)), 4) if cos else None}


def backgrounds(day, cam):
    """Recording-file backgrounds of a day and camera: [(start seconds of day, path)]."""
    out = []
    for p in (ROOT / 'data' / 'seg_datasets' / 'backgrounds').glob('%s_%s_*.jpg' % (day, cam)):
        hms, seg = p.stem.split('_')[2:4]
        t = int(hms[:2]) * 3600 + int(hms[2:4]) * 60 + int(hms[4:]) + 900 * int(seg)
        out.append((t, p))
    return sorted(out)


def test_reid(model, frames, dev, days=('20260917', '20260918', '20260919')):
    import door_review
    import slot_data
    res = {}
    for day in days:
        clean = json.load(open(ROOT / 'data' / 'door_review' / ('%s_clean.json' % day), encoding='utf-8'))['visits']
        events = {e['event_id']: e for e in door_review.day_events(day, ROOT)}
        ids = sorted({v['entry'] for v in clean} | {v['exit'] for v in clean})
        bgs = backgrounds(day, 'cam1')
        vec, miss = {}, 0
        for i in ids:
            e = events[i]
            lt = datetime.datetime.fromisoformat(e['time_local'])
            sec = lt.hour * 3600 + lt.minute * 60 + lt.second
            bgp = min(bgs, key=lambda b: abs(b[0] + 450 - sec))[1] if bgs else None      # the nearest file's background
            s = slot_data.load(frames, 'door_' + i, False, image=e['snapshot_full'], label=None, background=bgp, cam_day=(day, 'cam1'))
            out = run(model, s, dev)
            ppl = people(model, out)
            # the event's box: 2560x1440 -> 1280x720 -> the model's 1088x608 frame
            k = 0.5 * slot_data.W / slot_data.SW
            eb = [e['box_x1'] * k, e['box_y1'] * k, e['box_x2'] * k, e['box_y2'] * k]
            best, bi = None, 0.0
            for pp in ppl:
                cx, cy, w, h = pp[2]
                pb = [(cx - w / 2) * slot_data.W, (cy - h / 2) * slot_data.H, (cx + w / 2) * slot_data.W, (cy + h / 2) * slot_data.H]
                v = box_iou(eb, pb)
                if v > bi:
                    best, bi = pp, v
            if best is None or bi < 0.3:
                miss += 1
                vec[i] = None
                continue
            c = best[3] / np.linalg.norm(best[3]); sh = best[4] / np.linalg.norm(best[4])
            vec[i] = (c, sh)
        t = {i: events[i]['unix_ms'] / 1000 for i in ids}
        day_res = {'exits': len(clean), 'person_not_found': miss}
        for name, wc in MIXES.items():
            right = hard = hard_right = scored = 0
            for v in clean:
                if vec[v['exit']] is None or vec[v['entry']] is None:
                    continue
                inside = [w['entry'] for w in clean if t[w['entry']] < t[v['exit']] <= t[w['exit']] + 0.5 and vec[w['entry']] is not None]
                ex = vec[v['exit']]
                d = {e_: 1 - (wc * float(ex[0] @ vec[e_][0]) + (1 - wc) * float(ex[1] @ vec[e_][1])) for e_ in inside}
                best_e = min(d, key=d.get)
                scored += 1; right += best_e == v['entry']
                if len(inside) > 1:
                    hard += 1; hard_right += best_e == v['entry']
            day_res[name] = {'scored': scored, 'right_first': right, 'crowded': hard, 'crowded_right_first': hard_right}
        res[day] = day_res
    res['all'] = {name: {k: sum(res[d][name][k] for d in days) for k in ('scored', 'right_first', 'crowded', 'crowded_right_first')} for name in MIXES}
    res['all']['person_not_found'] = sum(res[d]['person_not_found'] for d in days)
    return res


def test_zone(model, frames, dev, held):
    import slot_data
    train = set(slot_data.train_ids(frames, 3))
    items = [(f, d) for f, d in frames.inout.items() if f not in train and f in frames.drafts and frames.image(f) is not None]
    names = ['outside', 'inside', 'doorway']
    conf = np.zeros((3, 3), int)
    miss = 0
    for ident, lab in items:
        s = slot_data.load(frames, ident, False)
        out = run(model, s, dev)
        ppl = people(model, out)
        vals = list(s['values'])
        for v, k in lab.items():
            if v not in vals:
                miss += 1; continue
            tm = s['masks'][vals.index(v)] > 0.5
            best, bi = None, 0.0
            for pp in ppl:
                inter = (pp[1] & tm).sum(); uni = (pp[1] | tm).sum()
                if uni and inter / uni > bi:
                    best, bi = pp, inter / uni
            if best is None or bi < 0.3:
                miss += 1; continue
            conf[k, int(np.argmax(best[5]))] += 1
    n = conf.sum()
    return {'people': int(n), 'not_found': miss, 'accuracy3': round(float(np.trace(conf) / max(1, n)), 4),
            'recall': {names[i]: round(float(conf[i, i] / max(1, conf[i].sum())), 3) for i in range(3)},
            'confusion_rows_truth': conf.tolist(),
            'inside_vs_outside_wrong': int(conf[0, 1] + conf[1, 0])}


def main():
    import torch
    import slot_data
    import eval_slots
    from slot_model import SlotModel
    ck_path = sys.argv[1]
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    frames = slot_data.Frames()
    ck = torch.load(ck_path, map_location='cpu')
    model = SlotModel(ck.get('backbone', 'deimv2_vit_tiny'), pretrained=False, mean=frames.mean, std=frames.std, teacher_dims=frames.teacher_dims)
    model.load_state_dict(ck.get('ema', ck['model']), strict=False)
    model = model.to(dev).eval()
    rep = {'checkpoint': ck_path, 'epoch': ck.get('epoch'), 'step': ck.get('step'), 'started': time.strftime('%H:%M:%S')}
    out = ROOT / 'data' / 'logs' / 'eval_slot_heads.json'
    save = lambda: json.dump(rep, open(out, 'w'), indent=1)
    held = eval_slots.held_out_ids()
    t0 = time.time(); rep['zone'] = test_zone(model, frames, dev, held); rep['zone']['s'] = round(time.time() - t0); save()
    t0 = time.time(); rep['reid'] = test_reid(model, frames, dev); rep['reid']['s'] = round(time.time() - t0); save()
    t0 = time.time()
    rep['radio_heldout'] = test_radio(model, frames, dev, held)
    tr = slot_data.train_ids(frames, 3)
    rep['radio_train'] = test_radio(model, frames, dev, tr[:: max(1, len(tr) // 200)][:200])
    rep['radio_s'] = round(time.time() - t0)
    rep['finished'] = time.strftime('%H:%M:%S'); save()


if __name__ == '__main__':
    main()
