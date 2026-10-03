"""The /inout crop classifier on the people of a door_v2 run: cut every tracked person out of the raw frame by the
box the run saved, as /inout did (1280 x 720 frame, the box +100 % wide, +25 % high, 96 x 160), and give the crop
and the box's place to model_cnn (section 37: 96.7 %, with the place boosting 97-98 %).

usage: door_io_crops.py RUN.jsonl.gz [EVERY]  -> RUN.io_cnn.json.gz {"<stretch>|<t>|<world>": [out, in, doorway]}
EVERY: classify every n-th tick (default 1)."""
import gzip
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))


def load_cnn(dev):
    import torch
    import inout_train as IT
    m = IT.net(False, 10, kind='cnn')
    res = m.load_state_dict(torch.load(ROOT / 'data' / 'inout' / 'model_cnn.pt', map_location='cpu'), strict=False)
    assert set(res.missing_keys) <= {'fmean', 'fstd'} and not res.unexpected_keys, res   # saved before the frame branch's constants
    return m.to(dev).eval()


def main(path, every=1):
    import torch
    import door_v2 as D
    import inout
    import inout_train as IT
    import sam31_segment as SS
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    m = load_cnn(dev)
    rows = []
    try:
        for l in gzip.open(path, 'rt'):
            rows.append(json.loads(l))
    except (EOFError, json.JSONDecodeError):
        pass
    head, ticks = rows[0], rows[1:]
    day = head['day']
    by_span = {}
    for r in ticks:
        by_span.setdefault(r['s'], []).append(r)
    cam = head.get('cam', 'cam1')
    raw = D.Raw(day, cam)
    dist = inout.floor_distance(cam)
    k = 1280 / 2176
    out, t0, n_done = {}, time.time(), 0
    status = Path(str(path).replace('.jsonl.gz', '.io_cnn.status.json'))
    for si, rs in sorted(by_span.items()):
        a, b = head['spans'][si]
        n = int(round((rs[-1]['t'] - a) / D.TICK)) + 1
        tks = SS.tick_frames(day, cam, a, n)
        X, G, keys = [], [], []
        for r in rs:
            i = int(round((r['t'] - a) / D.TICK))
            if i % every or not r['p']:
                continue
            img = raw.get(tks[i])
            if img is None:
                continue
            small = cv2.resize(img, (1280, 720), interpolation=cv2.INTER_AREA)
            for q in r['p']:
                cx, cy, w, h = q['box']
                x1, y1, x2, y2 = (cx - w / 2) * 2176 * k, (cy - h / 2) * 1248 * k, (cx + w / 2) * 2176 * k, (cy + h / 2) * 1248 * k
                X1, Y1, X2, Y2 = inout.crop_box([x1, y1, x2, y2])
                if X2 - X1 < 4 or Y2 - Y1 < 4:
                    continue
                X.append(cv2.resize(small[Y1:Y2, X1:X2], (IT.CW, IT.CH), interpolation=cv2.INTER_AREA)[:, :, ::-1])
                fx, fy = (q['foot'][0] * k, q['foot'][1] * k) if q.get('foot') else ((x1 + x2) / 2, y2)
                d = float(dist[int(np.clip(fy, 0, 719)), int(np.clip(fx, 0, 1279))])
                G.append([float(cam == 'cam2'), x1 / 1280, y1 / 720, x2 / 1280, y2 / 720, (x2 - x1) / 1280, (y2 - y1) / 720, fx / 1280, fy / 720, d / 100.0])
                keys.append('%d|%.2f|%d' % (si, r['t'], q['w']))
        if X:
            P = IT.predict_net(m, np.stack(X), np.zeros((len(X), 108, 192, 4), np.uint8), np.array(G, np.float32))
            for key, p in zip(keys, P):
                out[key] = [round(float(v), 4) for v in p]
        n_done += len(rs)
        json.dump({'span': si, 'spans': len(by_span), 'ticks': n_done, 'of': len(ticks), 'elapsed_s': round(time.time() - t0),
                   'stamp_mismatch': raw.off, 'updated': time.strftime('%H:%M:%S')}, open(status, 'w'))
    with gzip.open(str(path).replace('.jsonl.gz', '.io_cnn.json.gz'), 'wt') as f:
        json.dump(out, f)
    json.dump({'finished': time.strftime('%H:%M:%S'), 'people_ticks': len(out), 'elapsed_s': round(time.time() - t0),
               'stamp_mismatch': raw.off}, open(status, 'w'))


if __name__ == '__main__':
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 1)
