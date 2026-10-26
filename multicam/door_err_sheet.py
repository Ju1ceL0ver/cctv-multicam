"""Every remaining error of a door setting, as a picture (08.10.2026): the owner's /doorside crossings against the system's
on the small SAM's tracks; for each missed and each false crossing, 5 frames around it (-2..+2 s) with the tracks near the
door drawn (green inside / magenta outside / yellow unsure by the vote) and the drawn shop line.
-> data/door_v2/err_<src>/<n>_<miss|false>_<kind>_<tag>_<t>.jpg + index.json

usage: RA_BIN_SRC=micro1s3 RA_ZONE_STRIDE=3 door_err_sheet.py   (CPU)"""
import json
import os
import sys
from pathlib import Path

os.environ.setdefault('RA_BIN_SRC', 'micro1s3')
os.environ.setdefault('RA_BIN_VER', 'ldpc2')
os.environ.setdefault('RA_POS_VER', 'pc')
import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
OUT = ROOT / 'data' / 'door_v2'


def main():
    import cv2
    import door_combo as C
    import door_compare_video as V
    import door_learn as L
    import door_line
    import door_side_tune as ST
    src = os.environ['RA_BIN_SRC']
    stride = int(os.environ.get('RA_ZONE_STRIDE', '3'))
    cfg = json.loads(os.environ.get('RA_ERR_CFG', '{"mode": "either_in+both_out", "thr": 0.9, "h_in": 0, "h_out": 0, '
                                                  '"trunc": 0.0, "legs": "none", "conf": 2, "band": 100, "near": null, "birth": false}'))
    days, line, table = ST.load()
    f_ = json.load(open(OUT / 'combo_final.json'))
    u, hz = np.array(f_['u']), np.array(f_['zone'])
    out = OUT / ('err_%s' % src)
    out.mkdir(parents=True, exist_ok=True)
    index, n = [], 0
    for day, v in days.items():
        for tag, xs in v['obs'].items():
            a = v['dd'].S[tag]['a']
            per, vote_at = {}, {}
            for o in xs:
                if o['k'] % stride:
                    continue
                xy = v['dd'].P.get('%.2f_%d' % (o['t'], o['pid']))
                vt = door_line.side_vote(cfg, table, line, o['d'], o['top'], o['h'], o['bx'], o['p'])
                per.setdefault(o['pid'], []).append((o['t'], vt, xy))
                vote_at[(round(o['t'], 2), o['pid'])] = (vt, o['d'], o['p'])
            per = {k: sorted(x, key=lambda z: z[0]) for k, x in per.items()}
            roots = {k: k for k in per}
            tr = C.stitch_tracks(per, tuple(f_['stitch']), roots)
            ev = C.gate(C.flip_events({str(k): x for k, x in tr.items()}, cfg['conf'], 0.95), u, hz, f_['move'], f_['rad'])
            if cfg.get('birth'):
                ev = ev + C.birth_death({str(k): x for k, x in tr.items()}, cfg['conf'], 0.95, hz, f_['rad'], True, False, u=u,
                                        move=f_['move'], min_len=2.0, starts=[a], gain=0.05)
            ev = C.alternate(C.near_line(ev, line, cfg.get('near')), 'first')
            info = json.load(open(ROOT / 'data' / 'sam31_door' / tag / 'cam1' / 'info.json'))
            t_end = a + int(info['ticks']) * 0.08
            truth = [t for t in v['truth'] if a - 1 <= t['t'] - L.SHIFT <= t_end + 1]
            pred = [dict(e, t=e['t'] + L.SHIFT) for e in ev]
            used, hits = set(), set()
            for i, t in enumerate(truth):
                c = [(abs(p['t'] - t['t']), j) for j, p in enumerate(pred) if j not in used and p['kind'] == t['kind'] and abs(p['t'] - t['t']) <= L.TOL]
                if c:
                    j = min(c)[1]
                    used.add(j); hits.add(i)
            errs = [('miss', t['kind'], t['t'] - L.SHIFT) for i, t in enumerate(truth) if i not in hits] + \
                   [('false', p['kind'], p['t'] - L.SHIFT) for j, p in enumerate(pred) if j not in used]
            if not errs:
                continue
            base = ROOT / 'data' / 'micro_door' / src / tag / 'cam1'
            M, by_tick, binfo = V.people(base)
            ss = int(binfo.get('stride', 1))
            cap = cv2.VideoCapture(str(ROOT / 'data' / 'sam31_door' / tag / 'cam1' / 'video.mp4'))
            nfr = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            for what, kind, t in errs:
                tiles = []
                for dt in (-2.0, -1.0, 0.0, 1.0, 2.0):
                    fk = int(round((t + dt - a) / 0.08))
                    fk = max(0, min(nfr - 1, fk - fk % ss))
                    cap.set(cv2.CAP_PROP_POS_FRAMES, fk)
                    ok, fr = cap.read()
                    if not ok:
                        continue
                    img = cv2.resize(fr, (640, 360))
                    sx, sy = 640 / fr.shape[1], 360 / fr.shape[0]
                    for r, pid in by_tick.get(fk // ss, []):
                        x1, y1 = int(M.rows[r, 4]), int(M.rows[r, 5])
                        c_ = M.crop(r).astype(np.uint8)
                        m = np.zeros(fr.shape[:2], np.uint8)
                        m[y1:y1 + c_.shape[0], x1:x1 + c_.shape[1]] = c_[:fr.shape[0] - y1, :fr.shape[1] - x1]
                        m = cv2.resize(m, (640, 360), interpolation=cv2.INTER_NEAREST)
                        vt = vote_at.get((round(a + fk * 0.08, 2), pid), (None, None, None))
                        col = (0, 200, 0) if vt[0] == 1.0 else ((200, 0, 200) if vt[0] == 0.0 else (0, 220, 220))
                        cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        cv2.drawContours(img, cs, -1, col, 2)
                        if cs:
                            ys, xs_ = np.nonzero(m)
                            lab = '%d d%s p%s' % (pid % 1000, '' if vt[1] is None else int(vt[1]), '' if vt[2] is None else '%.2f' % vt[2])
                            cv2.putText(img, lab, (int(xs_.mean()) - 30, max(12, int(ys.min()) - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 3)
                            cv2.putText(img, lab, (int(xs_.mean()) - 30, max(12, int(ys.min()) - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, col, 1)
                    p1, p2 = line['p1'], line['p2']
                    cv2.line(img, (int(p1[0] / 2), int(p1[1] / 2)), (int(p2[0] / 2), int(p2[1] / 2)), (255, 255, 0), 1)
                    cv2.putText(img, '%+.0fs' % dt, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    tiles.append(img)
                if not tiles:
                    continue
                sheet = np.hstack(tiles)
                head = np.zeros((40, sheet.shape[1], 3), np.uint8)
                cv2.putText(head, '%s %s %s  %s  t=%.1f s from the stretch start' % (what.upper(), kind, day, tag, t - a), (10, 28),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                n += 1
                name = '%02d_%s_%s_%s_%d.jpg' % (n, what, kind, tag, int(t - a))
                cv2.imwrite(str(out / name), np.vstack([head, sheet]), [cv2.IMWRITE_JPEG_QUALITY, 85])
                index.append({'file': name, 'what': what, 'kind': kind, 'day': day, 'tag': tag, 't': round(t - a, 1)})
            cap.release()
    json.dump(index, open(out / 'index.json', 'w'), indent=1)
    print(len(index), 'errors drawn ->', out, flush=True)


if __name__ == '__main__':
    main()
