"""Every person of every SAM 3.1 single-frame draft, scored by the keep/delete network (data/keep2/keepnet_keep2.pt).
Only the scores are written -- data/keep2/all_scores.json {frame: {value: [score, area_full, x1, y1, x2, y2]}} -- the thresholds are
applied later (decide.py), so they can move without a new run. People under MIN_FULL pixels are not scored (the size rule deletes them)."""
import json
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import torch

sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parent))
import keepnet as K
import keep2 as K2
import rate


def prep(item):
    fid, img_p, lab_p = item
    img, lab = cv2.imread(str(img_p)), cv2.imread(str(lab_p), cv2.IMREAD_UNCHANGED)
    if img is None or lab is None:
        return fid, []
    lab = lab[..., 0] if lab.ndim == 3 else lab
    img = cv2.resize(img, (1280, 720)) if img.shape[:2] != (720, 1280) else img
    lab = cv2.resize(lab, (1280, 720), interpolation=cv2.INTER_NEAREST) if lab.shape != (720, 1280) else lab
    cam = 'cam2' if '_cam2_' in fid else 'cam1'
    rows = []
    for v, area, box, foot in K2.people(lab):
        af = int(area * K2.SCALE)
        if af < K.MIN_FULL:
            continue
        rows.append({'id': '%s|%d' % (fid, v), 'x': K.person_crop(img, lab, v, box), 'n': K.numbers(cam, foot, af, box, 1280, 720), 'y': 0, 'meta': [af] + box})
    return fid, rows


def main():
    t0 = time.time()
    root = K.ROOT
    idx = rate.index(str(root))
    drafts = sorted(p for p in (root / 'data' / 'sam31_stills' / 'drafts').glob('*.png') if p.stem in idx)
    items = [(p.stem, idx[p.stem][0], p) for p in drafts]
    ck = torch.load(root / 'data' / 'keep2' / 'keepnet_keep2.pt', map_location='cpu')
    net = K.Net().to(K.dev)
    net.load_state_dict(ck['model'])
    out, buf, n = {}, [], 0
    print('frames %d' % len(items), flush=True)

    def flush():
        if not buf:
            return
        p = K.score(net, buf)
        for r, s in zip(buf, p):
            fid, v = r['id'].split('|')
            out.setdefault(fid, {})[v] = [round(float(s), 4)] + r['meta']
        buf.clear()
    with ThreadPoolExecutor(6) as ex:
        for k, (fid, rows) in enumerate(ex.map(prep, items)):
            buf.extend(rows)
            n += len(rows)
            if len(buf) >= 1500:
                flush()
            if k % 1000 == 0:
                print('%d frames, %d people, %.0f s' % (k, n, time.time() - t0), flush=True)
    flush()
    json.dump(out, open(root / 'data' / 'keep2' / 'all_scores.json', 'w'))
    print('done: %d frames, %d people, %.0f s' % (len(out), n, time.time() - t0), flush=True)


if __name__ == '__main__':
    main()
