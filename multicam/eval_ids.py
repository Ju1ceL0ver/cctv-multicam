"""Score an identity assignment against gt_identity.json.

For each real person: share of their detections carrying their single most
common output id (1.0 = one id for the whole clip) and how many ids they were
split into. For each output id: purity. An 'identity error' is an output id
whose detections span two real people with the minority above 5%."""
import json, numpy as np
from collections import Counter, defaultdict


def evaluate(label, gt_path='data/clip/gt_identity.json', verbose=True):
    gt = json.load(open(gt_path))
    per_person = defaultdict(Counter)
    per_id = defaultdict(Counter)
    unlabeled = 0
    for cam in ('cam1', 'cam2'):
        for k, g in gt[cam].items():
            pid = label[cam].get(int(k))
            if pid is None:
                unlabeled += 1
                per_person[g]['<none>'] += 1
                continue
            per_person[g][pid] += 1
            per_id[pid][g] += 1
    rows = {}
    for g in sorted(per_person):
        c = per_person[g]
        tot = sum(c.values())
        ids = [k for k in c if k != '<none>']
        main = max((v for k, v in c.items() if k != '<none>'), default=0)
        rows[g] = {'detections': tot, 'main_id_share': main / tot, 'ids': len(ids), 'unassigned': c['<none>'] / tot}
    errors = []
    for pid, c in per_id.items():
        tot = sum(c.values())
        if len(c) > 1:
            minor = tot - max(c.values())
            if minor / tot > 0.05:
                errors.append((pid, dict(c)))
    if verbose:
        for g, r in rows.items():
            if g.startswith('G') and g != 'G':
                continue
            print('  %s: %5d dets | one-id share %.2f | split into %2d ids | unassigned %.2f' %
                  (g, r['detections'], r['main_id_share'], r['ids'], r['unassigned']))
        print('  identity errors (ids mixing people):', len(errors), errors[:6])
    return rows, errors
