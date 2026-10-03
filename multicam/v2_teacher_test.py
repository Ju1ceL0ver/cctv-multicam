"""The student against its teacher (SAM 3.1) on the held-out day 23.09 -- one cold frame at a time, scored the
exam's way (a person is found when a slot's mask covers theirs at IoU >= 0.5, one to one; person-ness >= 0.3, a
second slot on the same person dropped). Whatever SAM calls a person is a person here, the advertising stand too:
the owner's decision of 29.09 -- the training and the test follow the same teacher.

  build   a fixed list of moments: half the busiest (most people, overlapping boxes), half any moment with
          somebody; per camera, never two within SPACE_S -> data/v2_test/<name>.json
  Test    reads those frames and targets once (kept in memory, the frames as JPEG) and scores a model in ~20 s

usage: v2_teacher_test.py build NAME N TAG [TAG ...]
       v2_teacher_test.py run NAME CHECKPOINT"""
import json
import random
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
OUT = ROOT / 'data' / 'v2_test'
SPACE_S = 10.0
SMALL_PX = 204          # box height at 2176 wide = 120 px at 1280 (the exam's "small")
SCRAP_TICKS = 25        # a SAM person seen less than 2 s in the whole window ...
SCRAP_CELLS = 60        # ... or smaller than this (stride-4 cells) is a scrap: the "clean" score neither asks for it nor
                        # blames a slot on it (legs, a blob by the fan, half a person behind the stand -- tt_misses.jpg)
MIN_CELLS = 20          # a predicted mask smaller than this (stride-4 cells) is a speck, not an answer


def build(name, n, tags, seed=0):
    import v2_data as VD
    rng = random.Random(seed)
    per_cam = n // 2
    items = []
    for cam in ('cam1', 'cam2'):
        cands = []
        for tag in tags:
            w = VD.Window(tag, cam)
            ticks, counts = np.unique(w.t['tick'], return_counts=True)
            for t, c in zip(ticks.tolist(), counts.tolist()):
                rows = w.rows_at(t)
                b = w.t['box'][rows]
                over = 0
                for i in range(len(b)):
                    for j in range(i + 1, len(b)):
                        ix = max(0, min(b[i, 2], b[j, 2]) - max(b[i, 0], b[j, 0]))
                        iy = max(0, min(b[i, 3], b[j, 3]) - max(b[i, 1], b[j, 1]))
                        over += ix * iy > 0
                cands.append((c + over, rng.random(), tag, int(t)))
        taken = []

        def free(tag, t):
            return all(tg != tag or abs(t - tt) * VD.TICK >= SPACE_S for tg, tt in taken)
        busy = sorted(cands, key=lambda x: (-x[0], x[1]))
        for sc, _, tag, t in busy:
            if len(taken) >= per_cam // 2:
                break
            if free(tag, t):
                taken.append((tag, t))
        for sc, _, tag, t in sorted(cands, key=lambda x: x[1]):
            if len(taken) >= per_cam:
                break
            if free(tag, t):
                taken.append((tag, t))
        items += [{'tag': tag, 'cam': cam, 'tick': t, 'busy': k < per_cam // 2} for k, (tag, t) in enumerate(taken)]
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump({'tags': list(tags), 'items': items}, open(OUT / ('%s.json' % name), 'w'), indent=0)
    return items


def fix_items(verdict='take', kind='test'):
    """The owner's /fix frames (data/fix): his masks where he answered `verdict`, as (item, label map path)."""
    d = ROOT / 'data' / 'fix'
    m = json.load(open(d / 'manifest.json', encoding='utf-8'))
    st = json.load(open(d / 'state.json', encoding='utf-8')) if (d / 'state.json').exists() else {}
    out = []
    for it in m['items']:
        if it['set'] != kind or st.get(it['id'], {}).get('verdict') != verdict:
            continue
        lab = d / ('%s_mask.png' % it['id'])
        out.append((it, lab if lab.exists() else d / ('%s_init.png' % it['id'])))
    return out


class Test:
    def __init__(self, name):
        import v2_data as VD
        if name == 'fix':
            return self._fix()
        spec = json.load(open(OUT / ('%s.json' % name)))
        self.name, self.tags = name, spec['tags']
        self.items = []
        wins = {}
        for it in spec['items']:
            k = (it['tag'], it['cam'])
            if k not in wins:
                wins[k] = VD.Window(*k)
            w = wins[k]
            img = w.frame(it['tick'])
            if img is None:
                continue
            tg = w.targets(it['tick'])
            ok, jpg = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 95])
            h = tg['boxes'][:, 3] * VD.PH
            seen = np.array([len(w.track.get(int(p), ())) for p in tg['person']], np.int64)
            cells = tg['masks'].reshape(len(h), -1).sum(1)
            self.items.append({'scrap': (seen < SCRAP_TICKS) | (cells < SCRAP_CELLS), 'cam': it['cam'], 'busy': it.get('busy', False), 'jpg': jpg, 'bg_long': w.bg_long(it['tick']),
                               'bg_now': np.array(w.bg_now(it['tick'])), 'masks': np.packbits(tg['masks'].reshape(len(h), -1), 1),
                               'shape': tg['masks'].shape[1:], 'zone': tg['zone'], 'small': h < SMALL_PX})
        for w in wins.values():
            if w.cap is not None:
                w.cap.release()
                w.cap = None
        self.people = sum(len(x['zone']) for x in self.items)

    def _fix(self):
        """The test of the owner's eyes: the teacher test's moments he took on /fix, with his masks (the ones he threw
        out are left out). Zone: the SAM person a label came from (the draft's label k+1 is persons[k]), else none."""
        import v2_data as VD
        spec = {(i['tag'], i['cam'], int(i['tick'])): i for i in json.load(open(OUT / 'v2b.json'))['items']}
        self.name, self.items, wins = 'fix', [], {}
        for it, lab_p in fix_items('take', 'test'):
            k = (it['tag'], it['cam'])
            if k not in wins:
                wins[k] = VD.Window(*k)
            w = wins[k]
            img = cv2.imread(str(lab_p.parent / ('%s.jpg' % it['id'])))
            lab = cv2.imread(str(lab_p), cv2.IMREAD_UNCHANGED)
            if lab.ndim == 3:
                lab = lab[:, :, 0]
            rows = w.rows_at(it['tick'])
            zone_of = {int(w.t['person'][i]): int(w.t['zone'][i]) for i in rows}
            masks, zones, small = [], [], []
            for v in np.unique(lab):
                if not v:
                    continue
                full = lab == v
                g = cv2.resize(full.astype(np.float32), (VD.GW, VD.FH // 4), interpolation=cv2.INTER_AREA) >= 0.5
                if g.sum() < VD.MIN_PX:
                    continue
                m = np.zeros((VD.GH, VD.GW), bool)
                m[:VD.FH // 4] = g
                masks.append(m)
                p = it['persons'][v - 1] if v - 1 < len(it['persons']) else None
                zones.append(zone_of.get(p, -1) if p is not None else -1)
                ys = np.nonzero(full.any(1))[0]
                small.append(ys[-1] - ys[0] < SMALL_PX)
            n = len(masks)
            ok, jpg = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 95])
            G = np.stack(masks) if n else np.zeros((0, VD.GH, VD.GW), bool)
            self.items.append({'scrap': np.zeros(n, bool), 'cam': it['cam'], 'busy': spec.get((it['tag'], it['cam'], it['tick']), {}).get('busy', False),
                               'jpg': jpg, 'bg_long': w.bg_long(it['tick']), 'bg_now': np.array(w.bg_now(it['tick'])),
                               'masks': np.packbits(G.reshape(n, -1), 1), 'shape': (VD.GH, VD.GW), 'zone': np.array(zones, np.int64),
                               'small': np.array(small, bool)})
        for w in wins.values():
            if w.cap is not None:
                w.cap.release()
                w.cap = None
        self.people = sum(len(x['zone']) for x in self.items)

    def run(self, model, dev, thr=0.3):
        import torch
        import train_slots as TS
        import v2_data as VD
        import v2_eval
        from scipy.optimize import linear_sum_assignment
        asm = VD.Assembler(dev, getattr(model, 'rgb_size', None))
        was = model.training
        model.eval()
        rec = {'all': [0, 0], 'hall': [0, 0], 'out': [0, 0], 'door': [0, 0], 'small': [0, 0], 'busy': [0, 0], 'cam1': [0, 0], 'cam2': [0, 0]}
        found_any = false = 0
        clean_found = clean_n = clean_false = 0
        ious = []
        for x in self.items:
            img = cv2.imdecode(x['jpg'], cv2.IMREAD_COLOR)
            f = {'img': img, 'bg_long': x['bg_long'], 'bg_now': x['bg_now']}
            with torch.no_grad(), torch.autocast('cuda', torch.bfloat16, enabled=dev == 'cuda'):
                rgb, bgv, sta, cam_ids = asm([f], (x['cam'],), False)
                bg_sem = model.body.vit(bgv)[0]
                m = model.maps(rgb, sta, cam_ids, bg_sem, asm.last_world)
                r = model.decode(m, cam_ids)
                r, _ = model.cross_cameras(r, v2_eval.empty_like(r))
                logits = model.full_masks(m, r)[0]
                p = r['obj'][0].float().sigmoid().cpu().numpy()
                pad = r['pad'][0].cpu().numpy()
            P = (logits > 0).cpu().numpy()
            n = len(x['zone'])
            G = np.unpackbits(x['masks'], 1)[:, :int(np.prod(x['shape']))].astype(bool) if n else np.zeros((0, P[0].size), bool)
            P = P[:, :x['shape'][0], :x['shape'][1]].reshape(len(P), -1)
            ok = [s for s in range(len(p)) if not pad[s] and P[s].sum() >= MIN_CELLS]
            P, p = P[ok], p[ok]
            iou = np.zeros((n, len(P)), np.float32)
            if n and len(P):
                Gf, Pf = G.astype(np.float32), P.astype(np.float32)
                inter = Gf @ Pf.T
                iou = inter / np.maximum(Gf.sum(1)[:, None] + Pf.sum(1)[None] - inter, 1)
                found_any += int((iou.max(1) >= 0.5).sum())
            keep = [j for j in range(len(p)) if p[j] >= thr]
            keep = [keep[i] for i in TS.dedup([P[j] for j in keep], [p[j] for j in keep])]
            hit = {}
            if n and keep:
                rr, cc = linear_sum_assignment(-iou[:, keep])
                hit = {i: keep[j] for i, j in zip(rr, cc) if iou[i, keep[j]] >= 0.5}
            false += len(keep) - len(hit)
            scrap = x.get('scrap', np.zeros(n, bool))
            real = [i for i in range(n) if not scrap[i]]
            hit_c = {}
            if real and keep:
                rr, cc = linear_sum_assignment(-iou[np.ix_(real, keep)])
                hit_c = {real[i]: keep[j] for i, j in zip(rr, cc) if iou[real[i], keep[j]] >= 0.5}
            used = set(hit_c.values())
            for j in keep:
                if j not in used and not (n and scrap.any() and iou[scrap, j].max() >= 0.5):
                    clean_false += 1
            clean_found += len(hit_c); clean_n += len(real)
            ious += [float(iou[i, j]) for i, j in hit.items()]
            for i in range(n):
                got = int(i in hit)
                parts = ['all', x['cam']] + (['busy'] if x['busy'] else [])
                parts += {1: ['hall'], 0: ['out'], 2: ['door']}.get(int(x['zone'][i]), [])
                if x['small'][i]:
                    parts.append('small')
                for k in parts:
                    rec[k][0] += got; rec[k][1] += 1
        model.train(was)
        found, total = rec['all']
        R = found / max(1, total)
        Pr = found / max(1, found + false)
        out = {'frames': len(self.items), 'people': total, 'thr': thr, 'recall': round(R, 4), 'precision': round(Pr, 4),
               'f1': round(2 * R * Pr / max(1e-9, R + Pr), 4), 'false': false,
               'mask_iou_median': round(float(np.median(ious)), 4) if ious else None,
               'recall_any': round(found_any / max(1, total), 4)}
        Rc = clean_found / max(1, clean_n)
        Pc = clean_found / max(1, clean_found + clean_false)
        out.update(recall_clean=round(Rc, 4), precision_clean=round(Pc, 4), f1_clean=round(2 * Rc * Pc / max(1e-9, Rc + Pc), 4),
                   n_clean=clean_n, false_clean=clean_false)
        for k in ('hall', 'out', 'door', 'small', 'busy', 'cam1', 'cam2'):
            out['recall_' + k] = round(rec[k][0] / max(1, rec[k][1]), 4)
            out['n_' + k] = rec[k][1]
        return out


if __name__ == '__main__':
    if sys.argv[1] == 'build':
        its = build(sys.argv[2], int(sys.argv[3]), sys.argv[4:])
        print(len(its), 'moments', sum(i['busy'] for i in its), 'busy')
    elif sys.argv[1] == 'run':
        import torch
        import slot_v2 as V
        dev = 'cuda' if torch.cuda.is_available() else 'cpu'
        ck = torch.load(ROOT / sys.argv[3], map_location='cpu')
        m = V.build()
        m.load_state_dict(ck.get('ema', ck['model']))
        t = Test(sys.argv[2])
        print(t.people, 'people', flush=True)
        print(json.dumps(t.run(m.to(dev), dev)))
