"""The door as it would run on the camera (08.10.2026), on stretches the owner has not marked: the small SAM's people
(micro1s3), each coloured by the owner's inside/outside model (green inside, magenta outside, smoothed over 1 s),
and the system's ENTRY / EXIT the moment it decides (combo_final: side flip + movement + door zone, stitched tracks).
Nothing of the owner's answers is drawn: he checks it himself.

usage: door_sim_video.py DAY N_STRETCHES [OUT.mp4]  (CPU)"""
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
GREEN, MAGENTA, YELLOW = (0, 255, 0), (255, 0, 255), (0, 255, 255)


def main(day='20260919', n='3', out=None):
    import cv2
    os.environ['RA_BIN_SRC'] = 'micro1s3'
    import day_proxy
    import door_combo as C
    import door_compare_video as V
    import door_learn as L
    import door_mark as DM
    import door_side as S
    import door_v2 as D
    f = json.load(open(ROOT / 'data' / 'door_v2' / 'combo_final.json'))
    dd = C.Day(day)
    B = C.gate(dd.binary_events(f['conf'], f['thr'], tuple(f['stitch'])), np.array(f['u']), np.array(f['zone']), f['move'], f['rad'])
    st = S.load(day)
    marked = {t for t, s in st.items() if s.get('sides')}
    cand = [x for x in DM.stretches(day) if x['tag'] not in marked and x['tag'] in dd.S]
    cand.sort(key=lambda x: -len([a for a in x['answers'] if a['kind'] in ('in', 'out')]))
    pick = cand[:int(n)]
    out = out or str(ROOT / 'data' / 'logs' / ('door_sim_%s.mp4' % day))
    enc = subprocess.Popen([day_proxy._ffmpeg(), '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-s', '1280x720',
                            '-r', str(12.5 / 3), '-i', '-', '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '23', '-pix_fmt', 'yuv420p', out],
                           stdin=subprocess.PIPE)
    told = []
    for x in pick:
        tag = x['tag']
        a = x['a']
        mb = ROOT / 'data' / 'micro_door' / 'micro1s3' / tag / 'cam1'
        M, by_tick, info = V.people(mb)
        stride = int(info.get('stride', 3))
        obs = {}
        for t, pid, p in dd.S[tag]['obs']:
            obs.setdefault(pid, []).append((t, p))
        ev = [dict(e, ts=e['t'] - L.SHIFT - a) for e in B if e['w'].startswith(tag + ':')]
        told.append({'tag': tag, 'events': [(e['kind'], round(e['ts'], 1), e['w'].split(':')[-1]) for e in ev]})
        cap = cv2.VideoCapture(str(ROOT / 'data' / 'sam31_door' / tag / 'cam1' / 'video.mp4'))
        k = 0
        while True:
            ok, fr = cap.read()
            if not ok:
                break
            if k % stride == 0:
                ts = k * D.TICK
                img = cv2.resize(fr, (1280, 720))
                over = img.copy()
                sx, sy = 1280 / fr.shape[1], 720 / fr.shape[0]
                for r, pid in by_tick.get(k // stride, []):
                    xs = [p for t, p in obs.get(pid, []) if abs(t - (a + ts)) <= 0.5]
                    if not xs:
                        continue
                    p = float(np.mean(xs))
                    col = GREEN if p >= 0.95 else MAGENTA if p <= 0.05 else YELLOW
                    x1, y1 = int(M.rows[r, 4]), int(M.rows[r, 5])
                    m = np.zeros(fr.shape[:2], np.uint8)
                    c_ = M.crop(r)
                    m[y1:y1 + c_.shape[0], x1:x1 + c_.shape[1]] = c_[:fr.shape[0] - y1, :fr.shape[1] - x1]
                    m = cv2.resize(m, (1280, 720), interpolation=cv2.INTER_NEAREST).astype(bool)
                    if not m.any():
                        continue
                    over[m] = col
                    cs, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    cv2.drawContours(img, cs, -1, col, 5)
                    ys, xs_ = np.where(m)
                    tx, ty = int(xs_.mean()) - 30, max(30, int(ys.min()) - 10)
                    cv2.putText(img, 'P%d' % pid, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 9)
                    cv2.putText(img, 'P%d' % pid, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 1.3, col, 3)
                img = cv2.addWeighted(img, 0.6, over, 0.4, 0)
                on = [e for e in ev if 0 <= ts - e['ts'] <= 3.0]
                cv2.rectangle(img, (0, 0), (1280, 52), (0, 0, 0), -1)
                cv2.putText(img, '%s  %s  t=%.1f s   green = inside, magenta = outside, yellow = not sure' % (day, tag.split('_')[2], ts),
                            (12, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
                for i, e in enumerate(on):
                    txt = ('ENTRY  P%s' if e['kind'] == 'in' else 'EXIT  P%s') % e['w'].split(':')[-1]
                    col = GREEN if e['kind'] == 'in' else MAGENTA
                    y = 140 + 90 * i
                    cv2.putText(img, txt, (40, y), cv2.FONT_HERSHEY_SIMPLEX, 3.0, (0, 0, 0), 18)
                    cv2.putText(img, txt, (40, y), cv2.FONT_HERSHEY_SIMPLEX, 3.0, col, 7)
                enc.stdin.write(img.tobytes())
            k += 1
        cap.release()
    enc.stdin.close(); enc.wait()
    json.dump(told, open(Path(out).with_suffix('.json'), 'w'), indent=1)
    print(json.dumps(told), flush=True)


if __name__ == '__main__':
    main(*sys.argv[1:])
