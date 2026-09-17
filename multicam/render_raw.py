"""Review video for a raw clip: cam1 | cam2 | floor plan, one colour and number per person.

usage: render_raw.py CLIP FROM_S TO_S [out.mp4]"""
import sys, os, json, cv2, numpy as np
ROOT = os.path.dirname(os.path.abspath(__file__)); os.chdir(ROOT); sys.path.insert(0, ROOT)
from datetime import datetime, timedelta
from person3d import Camera, map_cam1
from fusion2 import build, fuse, visits, REG, DOOR, door_side
from clipdata import load
from rawsource import Stream, FPS
from topview import render

clip = sys.argv[1]; t0 = float(sys.argv[2]); t1 = float(sys.argv[3])
out_path = sys.argv[4] if len(sys.argv) > 4 else os.path.join('data', 'raw_clips', clip, 'review.mp4')
OUT_FPS = 10
dets, feats, meta, embs = load(clip)
raw_dets, _, _, _ = load(clip, apply_sync=False)
calib = json.load(open('data/calib_final.json'))
cams = {c: Camera(c, calib) for c in ('cam1', 'cam2')}
per_cam, _ = build(cams, dets, feats, embs, meta.get('clean'))
segs, people, _ = fuse(per_cam)
items = per_cam['cam1'] + per_cam['cam2']
label = {'cam1': {}, 'cam2': {}}
for pid, p in enumerate(people):
    for si in p['segments']:
        for m in segs[si]['members']:
            it = items[m]
            for di in it['det']:
                label[it['cam']][int(di)] = pid
V = {v['person']: v for v in visits(people)}
base = [(0, 90, 255), (255, 200, 0), (60, 220, 60), (255, 60, 200), (0, 230, 255), (200, 120, 255), (255, 255, 255),
        (40, 160, 255), (180, 255, 120), (120, 120, 255), (90, 200, 255), (255, 140, 90)]
colors = [base[i % len(base)] for i in range(len(people) + 1)]
ext = (-4.0, 6.5, -6.0, 12.0); res = 0.02
plan2, _, _ = render('cam2', calib, ext)
M = np.array(REG['M'])
plan1, _, _ = render('cam1', calib, ext, A=M[:, :2], T=M[:, 2])
plan = np.where((plan2.sum(2) == 0)[..., None], plan1, plan2)
plan = (plan * 0.5).astype(np.uint8)
to_px = lambda x, y: (int((x - ext[0]) / res), int((ext[3] - y) / res))
cv2.line(plan, to_px(*DOOR['door_a']), to_px(*DOOR['door_b']), (0, 220, 255), 8)
cv2.putText(plan, 'door', to_px(*DOOR['door_b']), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 220, 255), 3)
ph = 720; pw = int(plan.shape[1] * ph / plan.shape[0])
vw = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), OUT_FPS, (640 + pw, 720))
streams = {c: Stream(c, meta['day']) for c in ('cam1', 'cam2')}
start = datetime.fromisoformat(meta['start'])
for c, s_ in streams.items():
    s_.seek(start + timedelta(seconds=t0))
step = int(round(FPS / OUT_FPS))
off = meta.get('cam1_offset_s', 0.0)
k = int(t0 * FPS)
while k < int(t1 * FPS):
    frames = {}
    for c, s_ in streams.items():
        tt, fr = s_.read()
        if fr is None:
            break
        frames[c] = fr
    if len(frames) < 2:
        break
    if (k - int(t0 * FPS)) % step == 0:
        t_cam2 = k / FPS
        panes = []
        for c in ('cam1', 'cam2'):
            fr = frames[c]
            tt = t_cam2 - (off if c == 'cam1' else 0.0)
            d = raw_dets[c]
            for i in np.nonzero(np.abs(d[:, 0] - tt) < 0.021)[0]:
                pid = label[c].get(int(i))
                col = colors[pid] if pid is not None else (110, 110, 110)
                x1, y1, x2, y2 = d[i, 1:5].astype(int)
                cv2.rectangle(fr, (x1, y1), (x2, y2), col, 6 if pid is not None else 2)
                if pid is not None:
                    cv2.putText(fr, str(pid), (x1, max(50, y1 - 12)), cv2.FONT_HERSHEY_SIMPLEX, 2.4, col, 7)
            cv2.putText(fr, c, (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 2.6, (255, 255, 255), 7)
            panes.append(cv2.resize(fr, (640, 360)))
        pl = plan.copy()
        for pid, p in enumerate(people):
            past = p['t'] <= t_cam2 + 1e-6
            if past.sum() > 1:
                pts = np.array([to_px(*q) for q in p['xy'][past][-400:]], np.int32)
                cv2.polylines(pl, [pts], False, colors[pid], 4)
            now = np.nonzero(np.abs(p['t'] - t_cam2) < 0.06)[0]
            if len(now):
                cx, cy = to_px(*p['xy'][now[0]])
                cv2.circle(pl, (cx, cy), 16, colors[pid], -1)
                inside = door_side(p['xy'][now[0]].reshape(1, 2))[0] > 0
                cv2.putText(pl, '%d%s' % (pid, '' if inside else ' (снаружи)'), (cx + 18, cy - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.4, colors[pid], 4)
        pl = cv2.resize(pl, (pw, ph))
        cv2.putText(pl, 't=%.0f s' % t_cam2, (12, 36), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 2)
        vw.write(np.hstack([np.vstack(panes), pl]))
    k += 1
vw.release()
print('wrote', out_path, 'people', len(people))
