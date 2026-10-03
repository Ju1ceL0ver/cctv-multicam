"""How well a ReID model tells these shop's people apart, on things it was not taught on.

Two measures, both on 18.09:

* **door pairs** -- the owner's clean visits (data/door_review/<day>_clean.json): for every
  exit, the live counter's picture of the person leaving is compared with the entry
  pictures of everybody inside at that moment. Right first = the model would have closed
  the right visit. The entering picture is from the front and the leaving one from the back.
* **held-out tracks** -- the harvested pieces of that day: two pictures of one piece at
  least 3 s apart should be closer than two pieces the same camera showed at the same
  moment. AUC and the share of true pairs kept at a gate that joins 1 % of strangers.

A model trained on the same day's tracks has seen those people; its numbers are marked so.

usage: reid_eval_shop.py WEIGHTS [DAY]"""
import json
import os
import random
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))


def embedder(weights):
    import torch
    from boxmot.reid.core.reid import ReID
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    net = ReID(Path(weights), device=device, half=False)

    def embed(images):
        out = []
        for k in range(0, len(images), 64):
            chunk = images[k:k + 64]
            out.append(net(chunk))
        return np.concatenate(out) if out else np.zeros((0, 512), np.float32)
    return embed


def door_pairs(embed, day, root=ROOT):
    import cv2
    import door_review
    clean = json.load(open(Path(root) / 'data/door_review' / ('%s_clean.json' % day), encoding='utf-8'))['visits']
    events = {e['event_id']: e for e in door_review.day_events(day, root)}
    ids = sorted({v['entry'] for v in clean} | {v['exit'] for v in clean})
    pics = [cv2.imread(events[i]['snapshot_crop']) for i in ids]
    vec = dict(zip(ids, embed(pics)))
    t = {i: events[i]['unix_ms'] / 1000 for i in ids}
    right = hard = hard_right = 0
    margins = []
    for v in clean:
        inside = [w['entry'] for w in clean if t[w['entry']] < t[v['exit']] <= t[w['exit']] + 0.5]
        d = {e: 1 - float(vec[v['exit']] @ vec[e]) for e in inside}
        best = min(d, key=d.get)
        right += best == v['entry']
        if len(inside) > 1:
            hard += 1; hard_right += best == v['entry']
            others = [x for e, x in d.items() if e != v['entry']]
            margins.append(min(others) - d[v['entry']])
    return {'exits': len(clean), 'right_first': right, 'crowded': hard, 'crowded_right_first': hard_right,
            'crowded_margin_median': round(float(np.median(margins)), 4) if margins else None}


def held_out(embed, day, root=ROOT, pieces=1500, seed=0):
    import cv2
    from train_reid import load_people, rivals
    rng = random.Random(seed)
    people = load_people([day], root)
    riv = rivals(people)
    chosen = [i for i, r in enumerate(riv) if r]
    rng.shuffle(chosen)
    chosen = set(chosen[:pieces])
    for i in list(chosen):
        chosen.update(riv[i][:2])
    chosen = sorted(chosen)
    files, owner, when = [], [], []
    for i in chosen:
        p = people[i]
        for f in p['files'][:: max(1, len(p['files']) // 6)]:
            files.append(f); owner.append(i)
            when.append(p['first'] + (p['last'] - p['first']) * int(Path(f).stem) / max(1, len(p['files']) - 1))
    vec = embed([cv2.imread(f) for f in files])
    owner = np.array(owner); when = np.array(when)
    same, diff = [], []
    by = {}
    for k, i in enumerate(owner):
        by.setdefault(int(i), []).append(k)
    for i, ks in by.items():
        for a in range(len(ks)):
            for b in range(a + 1, len(ks)):
                if abs(when[ks[a]] - when[ks[b]]) >= 3:
                    same.append(1 - float(vec[ks[a]] @ vec[ks[b]]))
        for j in riv[i]:
            if j in by and j > i:
                for a in ks[:3]:
                    for b in by[j][:3]:
                        diff.append(1 - float(vec[a] @ vec[b]))
    same, diff = np.array(same), np.array(diff)
    gate = np.quantile(diff, 0.01)
    order = np.argsort(np.r_[same, diff])
    rank = np.empty(len(order)); rank[order] = np.arange(len(order))
    auc = 1 - (rank[:len(same)].sum() - len(same) * (len(same) - 1) / 2) / (len(same) * len(diff))
    return {'pieces': len(by), 'same_pairs': len(same), 'different_pairs': len(diff),
            'auc': round(float(auc), 4), 'kept_at_1pct': round(float((same <= gate).mean()), 4)}


def evaluate(weights, day='20260918', root=ROOT):
    embed = embedder(weights)
    return {'weights': str(weights), 'door': door_pairs(embed, day, root), 'held_out': held_out(embed, day, root)}


if __name__ == '__main__':
    os.chdir(ROOT)
    print(json.dumps(evaluate(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else '20260918'), indent=1))
