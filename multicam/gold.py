"""The owner's painted frames as ground truth: measure a segmenter on them, and train on them.

`eval MODEL [ids...]`   people found (IoU >= 0.5 with the owner's mask), false people, mask IoU,
                        split by size -- the number the student has to move
`export OUT ids...`     the frames as an Ultralytics segmentation set (1280x720, one polygon per
                        piece of each person's mask), to be joined with the earlier verified set

A person is one index of a label map; pieces smaller than MIN_PX are the specks nobody meant."""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
PAINT = ROOT / 'data' / 'paint'
MIN_PX = 60
SMALL = 120          # px of height in the 1280x720 frame: a far or a small person


def gold(ident):
    import cv2
    lab = cv2.imread(str(PAINT / ('%s_mask.png' % ident)), cv2.IMREAD_UNCHANGED)
    if lab.ndim == 3:                     # a one-channel PNG can come back as H x W x 1
        lab = lab[:, :, 0]
    return [lab == v for v in np.unique(lab) if v and (lab == v).sum() >= MIN_PX]


def done_ids():
    st = json.load(open(PAINT / 'state.json', encoding='utf-8'))
    return sorted(k for k, v in st.items() if v.get('done'))


def match(truth, pred):
    from scipy.optimize import linear_sum_assignment
    if not truth or not pred:
        return [], np.zeros((len(truth), len(pred)))
    iou = np.zeros((len(truth), len(pred)))
    for i, t in enumerate(truth):
        for j, p in enumerate(pred):
            inter = (t & p).sum()
            if inter:
                iou[i, j] = inter / (t | p).sum()
    r, c = linear_sum_assignment(-iou)
    return [(i, j) for i, j in zip(r, c) if iou[i, j] >= 0.5], iou


def evaluate(model_path, ids=None, imgsz=960, conf=0.25):
    import cv2
    from ultralytics import YOLO
    model = YOLO(str(model_path), task='segment')
    ids = ids or done_ids()
    found = total = false = 0
    small_found = small_total = 0
    ious = []
    for ident in ids:
        frame = cv2.imread(str(PAINT / ('%s.jpg' % ident)))
        truth = gold(ident)
        r = model.predict(frame, imgsz=imgsz, conf=conf, classes=[0], retina_masks=True, verbose=False)[0]
        pred = [np.squeeze(m).astype(bool) for m in r.masks.data.cpu().numpy()] if r.masks is not None else []
        pred = [m for m in pred if m.sum() >= MIN_PX]
        pairs, iou = match(truth, pred)
        total += len(truth); found += len(pairs); false += len(pred) - len(pairs)
        ious += [iou[i, j] for i, j in pairs]
        hit = {i for i, _ in pairs}
        for i, t in enumerate(truth):
            ys = np.nonzero(t.any(1))[0]
            if ys[-1] - ys[0] < SMALL:
                small_total += 1; small_found += i in hit
    return {'model': str(model_path), 'frames': len(ids), 'people': total, 'found': found,
            'recall': round(found / max(1, total), 4), 'false': false,
            'precision': round(found / max(1, found + false), 4),
            'mask_iou_median': round(float(np.median(ious)), 4) if ious else None,
            'small_people': small_total, 'small_recall': round(small_found / max(1, small_total), 4)}


def export(out, ids):
    """Frames and polygons in the Ultralytics layout; the caller writes data.yaml."""
    import cv2
    out = Path(out)
    for sub in ('images', 'labels'):
        (out / sub).mkdir(parents=True, exist_ok=True)
    n = 0
    for ident in ids:
        frame = cv2.imread(str(PAINT / ('%s.jpg' % ident)))
        h, w = frame.shape[:2]
        lines = []
        for m in gold(ident):
            cs, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for c in cs:
                if cv2.contourArea(c) < MIN_PX:
                    continue
                c = cv2.approxPolyDP(c, 1.0, True).reshape(-1, 2)
                if len(c) >= 3:
                    lines.append('0 ' + ' '.join('%.5f %.5f' % (x / w, y / h) for x, y in c))
        cv2.imwrite(str(out / 'images' / ('gold_%s.jpg' % ident)), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
        (out / 'labels' / ('gold_%s.txt' % ident)).write_text('\n'.join(lines))
        n += 1
    return n


if __name__ == '__main__':
    os.chdir(ROOT)
    if sys.argv[1] == 'eval':
        print(json.dumps(evaluate(sys.argv[2], sys.argv[3:] or None)))
    elif sys.argv[1] == 'export':
        print(export(sys.argv[2], sys.argv[3:]))
