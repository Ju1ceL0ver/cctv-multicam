"""Pictures for the review site: what a person looked like, big enough to judge.

Each piece gets a timeline strip (a few moments across its life) and one large portrait
from the frame where the person is biggest -- cut from the full 2560x1440 frame, not
from a halved copy, because the point of the site is that a face and a jacket are
actually visible. Everything outside the person's own outline is dimmed, so a neighbour
standing shoulder to shoulder is not mistaken for them.

usage: export_pieces.py CLIP [TAG]"""
import sys, os, json, cv2, numpy as np
ROOT = os.path.dirname(os.path.abspath(__file__)); os.chdir(ROOT); sys.path.insert(0, ROOT)
from person3d import Camera
from fusion2 import build
from clipdata import load
from rawsource import Stream
from datetime import datetime

clip = sys.argv[1]; tag = sys.argv[2] if len(sys.argv) > 2 else 'yolo26x-seg'
N = int(os.environ.get('RA_TILES', '6'))          # moments per strip
CH = int(os.environ.get('RA_TILE_H', '300'))      # pixel height of a strip tile
PH = int(os.environ.get('RA_PORTRAIT_H', '560'))  # pixel height of the big portrait

dets, feats, meta, embs = load(clip, tag)
raw_dets, _, _, _ = load(clip, tag, apply_sync=False)
polys = None
pp = os.path.join('data', 'raw_clips', clip, 'polys_%s.npz' % tag)
if os.path.exists(pp):
    z = dict(np.load(pp)); polys = {c: (z[c + '_pts'], z[c + '_off']) for c in ('cam1', 'cam2')}
pieces_file = os.path.join('data', 'raw_clips', clip, 'pieces_%s.json' % tag)
if os.path.exists(pieces_file):
    # Keep the pieces already being labelled; only their pictures are redrawn.
    from review_store import read_state
    old = read_state(os.path.dirname(pieces_file))['pieces']
    items = [{'cam': p['cam'], 't': dets[p['cam']][p['dets'], 0], 'xy': np.zeros((len(p['dets']), 2)),
              'det': np.array(p['dets']), 'keep_meta': p} for p in old]
else:
    cams = {c: Camera(c, json.load(open('data/calib_final.json'))) for c in ('cam1', 'cam2')}
    per_cam, _ = build(cams, dets, feats, embs, meta.get('clean'))
    items = sorted(per_cam['cam1'] + per_cam['cam2'], key=lambda it: it['t'][0])

# what to cut, keyed by the frame it lives in: one pass over the video, nothing held in memory
need = {'cam1': {}, 'cam2': {}}
for n, it in enumerate(items):
    d = it['det']
    sel = np.unique(np.linspace(0, len(d) - 1, N).astype(int))
    heights = raw_dets[it['cam']][d, 4] - raw_dets[it['cam']][d, 2]
    clean = meta.get('clean', {}).get(it['cam'], np.ones(len(raw_dets[it['cam']]), bool))[d]
    candidates = np.flatnonzero(clean)
    if not len(candidates): candidates = np.arange(len(d))
    best = int(candidates[np.argmax(heights[candidates])])
    for slot, pos in enumerate(sel):
        di = int(d[int(pos)])
        k = int(round(raw_dets[it['cam']][di, 0] * 25))
        need[it['cam']].setdefault(k, []).append((n, di, 'tile', slot))
    di = int(d[best])
    k = int(round(raw_dets[it['cam']][di, 0] * 25))
    need[it['cam']].setdefault(k, []).append((n, di, 'portrait', 0))

odir = os.path.join('data', 'raw_clips', clip, 'crops'); os.makedirs(odir, exist_ok=True)
pdir = os.path.join('data', 'raw_clips', clip, 'portraits'); os.makedirs(pdir, exist_ok=True)
tiles = {n: {} for n in range(len(items))}
portrait = {}


def cut(fr, cam, di, height):
    x1, y1, x2, y2 = raw_dets[cam][di, 1:5].astype(int)
    pad = int(0.12 * (y2 - y1)) + 6
    X1, Y1 = max(0, x1 - pad), max(0, y1 - pad)
    X2, Y2 = min(fr.shape[1], x2 + pad), min(fr.shape[0], y2 + pad)
    c = fr[Y1:Y2, X1:X2].copy()
    if c.size == 0:
        return None
    if polys is not None:
        pts, off = polys[cam]
        a, b = off[di], off[di + 1]
        if b > a:
            poly = (pts[a:b] - [X1, Y1]).astype(np.int32)
            m = np.zeros(c.shape[:2], np.uint8)
            cv2.fillPoly(m, [poly], 255)
            m = cv2.dilate(m, np.ones((5, 5), np.uint8))
            c = np.where(m[..., None] > 0, c, (c * 0.35).astype(np.uint8))
    w = max(24, int(c.shape[1] * height / max(1, c.shape[0])))
    return cv2.resize(c, (w, height), interpolation=cv2.INTER_CUBIC)


start = datetime.fromisoformat(meta['start'])
for cam in ('cam1', 'cam2'):
    if not need[cam]:
        continue
    s = Stream(cam, meta['day']); s.seek(start)
    last = max(need[cam])
    for k in range(last + 1):
        t, fr = s.read()
        if fr is None:
            break
        for n, di, kind, slot in need[cam].get(k, ()):
            c = cut(fr, cam, di, PH if kind == 'portrait' else CH)
            if c is None:
                continue
            if kind == 'portrait':
                portrait[n] = c
            else:
                cv2.putText(c, '%.0fs' % dets[cam][di, 0], (4, CH - 8), cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, (0, 240, 255), 2, cv2.LINE_AA)
                tiles[n][slot] = c

index = []
for n, it in enumerate(items):
    parts = [tiles[n][s] for s in sorted(tiles[n])] or [np.zeros((CH, 60, 3), np.uint8)]
    cv2.imwrite(os.path.join(odir, '%04d.jpg' % n), np.hstack(parts), [cv2.IMWRITE_JPEG_QUALITY, 92])
    if n in portrait:
        cv2.imwrite(os.path.join(pdir, '%04d.jpg' % n), portrait[n], [cv2.IMWRITE_JPEG_QUALITY, 92])
    if 'keep_meta' in it:
        index.append(it['keep_meta'])
    else:
        index.append({'piece': n, 'cam': it['cam'], 't0': round(float(it['t'][0]), 1), 't1': round(float(it['t'][-1]), 1),
                      'xy0': [round(float(v), 1) for v in it['xy'][0]], 'xy1': [round(float(v), 1) for v in it['xy'][-1]],
                      'dets': [int(x) for x in it['det']], 'n_dets': len(it['det'])})
json.dump(index, open(pieces_file, 'w'))
print('%s: %d pieces, tiles %dpx, portraits %dpx, cut from full frames' % (clip, len(items), CH, PH))
