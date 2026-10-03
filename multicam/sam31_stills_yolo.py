"""SAM 3.1's single-frame labels (sam31_stills.py) of the 13 thousand varied frames as a YOLO-seg set, the exam day
and the held-out day left out; the images are hard links to the draft frames (1280 x 720), not copies.

usage: sam31_stills_yolo.py   -> data/seg_datasets/sam31_stills_yolo/{images,labels}/train"""
import concurrent.futures as cf
import os
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
OUT = ROOT / 'data' / 'seg_datasets' / 'sam31_stills_yolo'
SKIP = ('20260918', '20260923')


def one(job):
    ident, img = job
    lab = cv2.imread(str(ROOT / 'data' / 'sam31_stills' / 'drafts' / ('%s.png' % ident)), cv2.IMREAD_UNCHANGED)
    if lab is None:
        return 0
    h, w = lab.shape[:2]
    lines = []
    for v in np.unique(lab):
        if not v:
            continue
        cs, _ = cv2.findContours((lab == v).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in cs:
            if cv2.contourArea(c) < 20:
                continue
            c = cv2.approxPolyDP(c, 1.0, True).reshape(-1, 2).astype(np.float32)
            if len(c) < 3:
                continue
            c[:, 0] /= w
            c[:, 1] /= h
            lines.append('0 ' + ' '.join('%.5f %.5f' % (x, y) for x, y in np.clip(c, 0, 1)))
    dst = OUT / 'images' / 'train' / ('%s.jpg' % ident)
    if not dst.exists():
        try:
            os.link(img, dst)
        except OSError:
            import shutil
            shutil.copy2(img, dst)
    (OUT / 'labels' / 'train' / ('%s.txt' % ident)).write_text('\n'.join(lines))
    return 1


def main():
    import rate
    for sub in ('images/train', 'labels/train'):
        (OUT / sub).mkdir(parents=True, exist_ok=True)
    jobs = [(i, str(p)) for i, (p, _) in rate.index(str(ROOT), force=True).items() if not i.startswith(SKIP)]
    with cf.ThreadPoolExecutor(12) as ex:
        n = sum(ex.map(one, jobs))
    print('sam31_stills_yolo:', n, 'frames', flush=True)


if __name__ == '__main__':
    main()
