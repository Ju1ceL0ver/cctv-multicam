"""15.10 owner: how good are the door model and the SAM 3.1 teacher on crowds, against his /crowdmasks masks.

Per labelled frame (data/live/crowd_labels/<day>/<key>/<i>.png, instance ids): the door model's masks of that tick
(data/live/crowd/.../masks.npz) and, when the night pass made it, the teacher's (data/live/crowd_teacher/.../<i>.png).
People are matched one to one by mask IoU >= 0.5; a merged mask is one that covers >= 30 % of two owner's people.
usage: crowd_compare.py [teacher]   ('teacher' first runs SAM 3.1 on the labelled frames that lack it; GPU, night)
-> data/live/crowd_compare.json and a printed table."""
import json, sys
from pathlib import Path
import numpy as np, cv2
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
LIVE = ROOT / 'data' / 'live'


def frames():
    for f in sorted((LIVE / 'crowd_labels').glob('*/*/*.json')):
        rec = json.loads(f.read_text(encoding='utf-8'))
        if rec.get('skip') or not f.with_suffix('.png').exists():
            continue
        yield rec, f.with_suffix('.png')


def owner_masks(png):
    ids = cv2.imread(str(png), cv2.IMREAD_UNCHANGED)
    return [ids == k for k in range(1, int(ids.max()) + 1) if (ids == k).any()]


def model_masks(rec):
    import crowd_page as C
    return [cv2.resize(m.astype(np.uint8), (1280, 720), interpolation=cv2.INTER_NEAREST) > 0
            for _, m in C.model_masks(LIVE / 'crowd' / rec['day'] / rec['key'], rec['i'])]


def teacher_masks(rec):
    p = LIVE / 'crowd_teacher' / rec['day'] / rec['key'] / ('%04d.png' % rec['i'])
    if not p.exists():
        return None
    ids = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
    return [ids == k for k in range(1, int(ids.max()) + 1) if (ids == k).any()]


def score(truth, pred):
    from scipy.optimize import linear_sum_assignment
    if not truth or not pred:
        return {'truth': len(truth), 'pred': len(pred), 'tp': 0, 'merged': 0, 'iou': []}
    I = np.array([[(a & b).sum() / max(1, (a | b).sum()) for b in pred] for a in truth])
    r, c = linear_sum_assignment(-I)
    hit = [(i, j) for i, j in zip(r, c) if I[i, j] >= .5]
    merged = sum(1 for b in pred if sum((a & b).sum() >= .3 * a.sum() for a in truth) >= 2)
    return {'truth': len(truth), 'pred': len(pred), 'tp': len(hit), 'merged': merged, 'iou': [float(I[i, j]) for i, j in hit]}


def run_teacher():
    import tempfile
    import sam31_segment as SG
    import sam31_stills as ST
    pred, folder = SG.build(), Path(tempfile.mkdtemp(prefix='crowdteach_'))
    for rec, _ in frames():
        out = LIVE / 'crowd_teacher' / rec['day'] / rec['key'] / ('%04d.png' % rec['i'])
        if out.exists():
            continue
        img = cv2.imread(str(LIVE / 'crowd' / rec['day'] / rec['key'] / ('%04d.jpg' % rec['i'])))
        got = sorted(ST.one(pred, img, folder), key=lambda x: -int(x[1].sum()))[:255]
        lab = np.zeros((720, 1280), np.uint8)
        for k, (_, m) in enumerate(got):
            lab[m] = k + 1
        out.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out), lab)


def main():
    if 'teacher' in sys.argv:
        run_teacher()
    tot = {'model': [], 'teacher': []}
    for rec, png in frames():
        truth = owner_masks(png)
        tot['model'].append(score(truth, model_masks(rec)))
        t = teacher_masks(rec)
        if t is not None:
            tot['teacher'].append(score(truth, t))
    out = {}
    for name, rows in tot.items():
        if not rows:
            continue
        T = sum(r['truth'] for r in rows); P = sum(r['pred'] for r in rows); TP = sum(r['tp'] for r in rows)
        ious = [x for r in rows for x in r['iou']]
        out[name] = {'frames': len(rows), 'people': T, 'recall': round(TP / max(1, T), 3), 'precision': round(TP / max(1, P), 3),
                     'merged_masks': sum(r['merged'] for r in rows), 'iou_median': round(float(np.median(ious)), 3) if ious else None,
                     'frames_all_right': sum(r['tp'] == r['truth'] == r['pred'] for r in rows)}
        print(name, out[name])
    json.dump(out, open(LIVE / 'crowd_compare.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
