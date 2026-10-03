"""The SAM 3.1 windows as a YOLO-seg dataset -- the yardstick of 30.09: an ordinary detector (yolo26s-seg) on the same
teacher and the same test as the slot model. If it goes well past the slot model, the slot model is the problem;
if it stops where the slot model stops, the teacher is the ceiling.

Train: every STEP-th tick of the training windows (both cameras), 1280 x 720 JPEG; a person = the largest outline of
their SAM mask (the stride-4 grid, as the slot model learns it). Val: the teacher test's moments (data/v2_test/v2b.json,
the held-out day 23.09) -- YOLO's own numbers only; the comparison is v2_teacher_test's score.

usage: yolo_sam_export.py [STEP]      -> data/seg_datasets/sam31_yolo/{images,labels}/{train,val}, data.yaml"""
import concurrent.futures as cf
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
OUT = ROOT / 'data' / 'seg_datasets' / 'sam31_yolo'
HELD = ('20260923',)


def polygons(masks):
    """grid masks (n x GH x GW) -> YOLO lines (class 0, the largest outline, normalised to the stored frame)."""
    import v2_data as VD
    h, w = VD.FH // 4, VD.GW
    out = []
    for m in masks:
        g = m[:h].astype(np.uint8)
        g = cv2.resize(g, (w * 2, h * 2), interpolation=cv2.INTER_NEAREST)      # a smoother outline
        cs, _ = cv2.findContours(g, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cs:
            continue
        c = max(cs, key=cv2.contourArea)
        if cv2.contourArea(c) < 8:
            continue
        c = cv2.approxPolyDP(c, 1.0, True).reshape(-1, 2).astype(np.float32)
        if len(c) < 3:
            continue
        c[:, 0] /= w * 2
        c[:, 1] /= h * 2
        out.append('0 ' + ' '.join('%.5f %.5f' % (x, y) for x, y in np.clip(c, 0, 1)))
    return out


def one(tag, cam, ticks, split):
    import v2_data as VD
    w = VD.Window(tag, cam)
    n = 0
    for t in sorted(ticks):
        img = w.frame(t)
        if img is None:
            continue
        tg = w.targets(t)
        name = '%s_%s_%05d' % (tag, cam, t)
        cv2.imwrite(str(OUT / 'images' / split / (name + '.jpg')), cv2.resize(img, (1280, 720), interpolation=cv2.INTER_AREA),
                    [cv2.IMWRITE_JPEG_QUALITY, 90])
        (OUT / 'labels' / split / (name + '.txt')).write_text('\n'.join(polygons(tg['masks'])))
        n += 1
    if w.cap is not None:
        w.cap.release()
    return tag, cam, n


def main():
    import v2_data as VD
    step = int(sys.argv[1]) if len(sys.argv) > 1 else 25
    t0 = time.time()
    for s in ('train', 'val'):
        (OUT / 'images' / s).mkdir(parents=True, exist_ok=True)
        (OUT / 'labels' / s).mkdir(parents=True, exist_ok=True)
    jobs = []
    for tag, cam in VD.window_cams(exclude=('20260918',)):
        if tag.startswith(HELD):
            continue
        n = json.load(open(VD.V2 / tag / cam / 'meta.json'))['ticks']
        jobs.append((tag, cam, list(range(0, n, step)), 'train'))
    val = {}
    for it in json.load(open(ROOT / 'data' / 'v2_test' / 'v2b.json'))['items']:
        val.setdefault((it['tag'], it['cam']), []).append(it['tick'])
    jobs += [(tag, cam, ticks, 'val') for (tag, cam), ticks in val.items()]
    with cf.ProcessPoolExecutor(6) as ex:
        for tag, cam, n in ex.map(one, *zip(*jobs)):
            print(tag, cam, n, 'frames', round(time.time() - t0), 's', flush=True)
    (OUT / 'data.yaml').write_text('path: %s\ntrain: images/train\nval: images/val\nnames:\n  0: person\n' % OUT.as_posix())
    print('done: %d train, %d val, %.0f s' % (len(list((OUT / 'images' / 'train').glob('*.jpg'))),
                                                len(list((OUT / 'images' / 'val').glob('*.jpg'))), time.time() - t0), flush=True)


if __name__ == '__main__':
    main()
