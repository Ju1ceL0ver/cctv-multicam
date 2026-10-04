"""A short clip for the owner: SAM 3.1 masks at the door, each person's side, the counted entries and exits (04.10.2026).

usage: door_video.py DAY FROM_S TO_S [OUT.mp4]   (film seconds; the stretch is a door_sam stretch that holds them)"""
import gzip
import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
COL = [(66, 197, 245), (80, 220, 100), (245, 160, 66), (220, 90, 220), (240, 220, 70), (160, 100, 255), (60, 200, 200),
       (200, 160, 90), (120, 230, 180), (250, 120, 160), (90, 140, 250), (230, 230, 230)]
SIDE = ['OUT', 'IN', 'DOOR']


def main(day, a, b, out=None):
    import day_proxy
    import door_learn as L
    import door_sam as DS
    import door_v2 as D
    import sam31_reid as R
    a, b = float(a), float(b)
    spans = D.stretches(day)
    si = next(i for i, (x, y) in enumerate(spans) if x <= a and b <= y)
    sa = spans[si][0]
    base = DS.DOOR / DS.tag_of(day, sa) / DS.CAM
    M = R.Masks(base / 'chunks.npz')
    info = json.load(open(base / 'info.json'))
    rep = json.load(open(base / 'report.json'))
    owned, _ = R.link_seams(M, {int(s): int(sh) for s, e, sh in info['sessions']})
    person = {int(p): v for p, v in rep['person_of_piece'].items()}
    # the advertising stand by the door: SAM calls it a person; it never moves -- drop such "people"
    import door_sam as DS2
    static = DS2.static_people(M, owned, person, info['ticks'])
    rows_of = {}
    for p, rs in owned.items():
        if (person.get(int(p)) or 0) in static:
            continue
        for r in rs:
            rows_of.setdefault(int(M.rows[r, 1]), []).append((r, person.get(int(p)) or 0))
    # sides from the SAM run (place + crop classifiers), and the counted crossings
    run = ROOT / 'data' / 'door_v2' / ('%s_sam31.jsonl.gz' % day)
    ticks, _ = D._read(run)
    ticks = [r for r in ticks if r['s'] == si]
    D.add_io(ticks, run)
    side = {}
    for r in ticks:
        for q in r['p']:
            side[(round(r['t'], 2), q['w'] % 10000)] = int(np.argmax(q.get('io_mix', q['io'])))
    preds = json.load(open(ROOT / 'data' / 'door_v2' / ('learn_preds_%s_sam.json' % day)))
    thr = json.load(open(ROOT / 'data' / 'door_v2' / 'learn.json'))['days'].get(day, {}).get('threshold', 0.5)
    events = [(p['t'] - L.SHIFT, p['kind']) for p in preds if p['p'] >= thr and a - 1 <= p['t'] - L.SHIFT <= b]
    truth = [(t['t'] - L.SHIFT, t['kind']) for t in D.truth(day, True) if a - 1 <= t['t'] - L.SHIFT <= b and t['kind'] in ('in', 'out')]
    out = out or str(ROOT / 'data' / 'door_v2' / ('door_clip_%s_%d.mp4' % (day, a)))
    W, H = 1280, 720
    enc = subprocess.Popen([day_proxy._ffmpeg(), '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-s', '%dx%d' % (W, H),
                            '-r', '12.5', '-i', '-', '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '23', '-pix_fmt', 'yuv420p', out],
                           stdin=subprocess.PIPE)
    cap = cv2.VideoCapture(str(base / 'video.mp4'))
    k0, k1 = int(round((a - sa) / D.TICK)), int(round((b - sa) / D.TICK))
    cap.set(cv2.CAP_PROP_POS_FRAMES, k0)
    for k in range(k0, k1):
        ok, img = cap.read()
        if not ok:
            break
        t = round(sa + k * D.TICK, 2)
        over = img.copy()
        labels = []
        for r, pid in rows_of.get(k, []):
            x1, y1, x2, y2 = M.rows[r, 4:8].astype(int)
            m = np.zeros(img.shape[:2], bool)
            m[y1:y2, x1:x2] = M.crop(r)
            col = COL[pid % len(COL)]
            over[m] = col
            cs, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(img, cs, -1, col, 4)
            s = side.get((t, pid))
            ly = y1 - 12 if y1 - 12 > 120 else y2 + 45          # under the person when the top is behind the title bar
            labels.append(((x1 + x2) // 2, ly, 'P%d %s' % (pid, SIDE[s] if s is not None else ''), col))
        img = cv2.addWeighted(img, 0.65, over, 0.35, 0)
        for x, y, txt, col in labels:
            cv2.putText(img, txt, (x - 60, y), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 0, 0), 8)
            cv2.putText(img, txt, (x - 60, y), cv2.FONT_HERSHEY_SIMPLEX, 1.4, col, 3)
        cv2.rectangle(img, (0, 0), (img.shape[1], 70), (0, 0, 0), -1)
        now = sa + k * D.TICK
        ev = [kind for te, kind in events if 0 <= now - te <= 2.0]
        gt = [kind for te, kind in truth if 0 <= now - te <= 2.0]
        msg = '19.09 door  t=%.1f s' % now if day == '20260919' else '%s door t=%.1f' % (day, now)
        cv2.putText(img, msg, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (255, 255, 255), 3)
        if ev:
            cv2.putText(img, 'MODEL: ' + ' + '.join('ENTRY' if e == 'in' else 'EXIT' for e in ev), (700, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (80, 255, 80), 4)
        if gt:
            cv2.putText(img, 'OWNER: ' + ' + '.join('ENTRY' if e == 'in' else 'EXIT' for e in gt), (1400, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (80, 200, 255), 4)
        enc.stdin.write(cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA).tobytes())
    cap.release(); enc.stdin.close(); enc.wait()
    print(out, len(events), 'model events', len(truth), 'owner events')


if __name__ == '__main__':
    main(*sys.argv[1:])
