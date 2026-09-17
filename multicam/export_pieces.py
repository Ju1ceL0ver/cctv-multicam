"""Per-piece crop strips for manual identity labelling.

One image per tracker piece: 8 crops across its life, cut by the person's own
segmentation mask (the neighbour standing shoulder to shoulder is erased), so
what the eye compares is the person, not the pair.

usage: export_pieces.py CLIP [TAG]"""
import sys, os, json, cv2, numpy as np
ROOT = os.path.dirname(os.path.abspath(__file__)); os.chdir(ROOT); sys.path.insert(0, ROOT)
from person3d import Camera
from fusion2 import build
from clipdata import load, frames_needed

clip = sys.argv[1]; tag = sys.argv[2] if len(sys.argv) > 2 else 'yolo26x-seg'
N, CH = 8, 190
dets, feats, meta, embs = load(clip, tag)
raw_dets, _, _, _ = load(clip, tag, apply_sync=False)
polys = None
pp = os.path.join('data', 'raw_clips', clip, 'polys_%s.npz' % tag)
if os.path.exists(pp):
    z = dict(np.load(pp)); polys = {c: (z[c + '_pts'], z[c + '_off']) for c in ('cam1', 'cam2')}
pieces_file = os.path.join('data', 'raw_clips', clip, 'pieces_%s.json' % tag)
if os.path.exists(pieces_file):
    # Keep the pieces already being labelled; only their pictures are redrawn.
    old = json.load(open(pieces_file))
    items = [{'cam': p['cam'], 't': dets[p['cam']][p['dets'], 0], 'xy': np.zeros((len(p['dets']), 2)),
              'det': np.array(p['dets']), 'keep_meta': p} for p in old]
else:
    cams = {c: Camera(c, json.load(open('data/calib_final.json'))) for c in ('cam1', 'cam2')}
    per_cam, _ = build(cams, dets, feats, embs, meta.get('clean'))
    items = sorted(per_cam['cam1'] + per_cam['cam2'], key=lambda it: it['t'][0])
wanted = {'cam1': set(), 'cam2': set()}
picks = []
for it in items:
    sel = np.unique(np.linspace(0, len(it['det']) - 1, N).astype(int))
    ks = [(int(it['det'][p]), int(round(raw_dets[it['cam']][int(it['det'][p]), 0] * 25))) for p in sel]
    for _, k in ks:
        wanted[it['cam']].add(k)
    picks.append(ks)
frames = frames_needed(meta, wanted)
odir = os.path.join('data', 'raw_clips', clip, 'crops'); os.makedirs(odir, exist_ok=True)
index = []
for n, (it, ks) in enumerate(zip(items, picks)):
    tiles = []
    for di, k in ks:
        fr = frames[it['cam']].get(k)
        if fr is None:
            continue
        x1, y1, x2, y2 = (raw_dets[it['cam']][di, 1:5] / 2).astype(int)
        pad = int(0.1 * (y2 - y1))
        X1, Y1 = max(0, x1 - pad), max(0, y1 - pad); X2, Y2 = min(1280, x2 + pad), min(720, y2 + pad)
        c = fr[Y1:Y2, X1:X2].copy()
        if c.size == 0:
            continue
        if polys is not None:
            pts, off = polys[it['cam']]
            a, b = off[di], off[di + 1]
            if b > a:
                poly = (pts[a:b] / 2 - [X1, Y1]).astype(np.int32)
                m = np.zeros(c.shape[:2], np.uint8)
                cv2.fillPoly(m, [poly], 255)
                m = cv2.dilate(m, np.ones((3, 3), np.uint8))
                c = np.where(m[..., None] > 0, c, (c * 0.25).astype(np.uint8))
        c = cv2.resize(c, (max(30, int(c.shape[1] * CH / c.shape[0])), CH))
        cv2.putText(c, '%.0fs' % dets[it['cam']][di, 0], (3, CH - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        tiles.append(c)
    if not tiles:
        tiles = [np.zeros((CH, 60, 3), np.uint8)]
    strip = np.hstack(tiles)
    cv2.imwrite(os.path.join(odir, '%04d.jpg' % n), strip, [cv2.IMWRITE_JPEG_QUALITY, 88])
    if 'keep_meta' in it:
        index.append(it['keep_meta'])
    else:
        index.append({'piece': n, 'cam': it['cam'], 't0': round(float(it['t'][0]), 1), 't1': round(float(it['t'][-1]), 1),
                      'xy0': [round(float(v), 1) for v in it['xy'][0]], 'xy1': [round(float(v), 1) for v in it['xy'][-1]],
                      'dets': [int(x) for x in it['det']], 'n_dets': len(it['det'])})
json.dump(index, open(os.path.join('data', 'raw_clips', clip, 'pieces_%s.json' % tag), 'w'))
print(clip, 'pieces', len(items), 'masked crops' if polys is not None else 'box crops (no polygons yet)')
