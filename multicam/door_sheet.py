"""Sheets of the door rule's errors to look at (03.10.2026): for each error five raw frames of camera 1 around it
(-3, -1.5, 0, +1.5, +3 s), every tracked person boxed with the side the /inout classifiers give (o / i / d), the
person of the error in red, the owner's answers within 10 s written on top.

usage: door_sheet.py DAY RUN.jsonl.gz [ERRORS.json]  -> data/door_v2/sheets/<day>_NN.jpg"""
import gzip
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
OFFS = (-3.0, -1.5, 0.0, 1.5, 3.0)
TW, TH = 544, 306


def main(day, path, errors_p=None):
    import door_v2 as D
    import sam31_segment as SS
    import os
    if errors_p == 'truth':                       # the owner's crossings themselves, to see how a camera shows them
        import door_v2 as D0
        import door_learn as L0
        _, sp = D0._read(path)
        errs = [{'day': day, 'what': 'truth', 'kind': t['kind'], 't': t['t'] - L0.SHIFT, 'w': None}
                for t in D0.truth(day, True) if any(a <= t['t'] <= b for a, b in sp)][:int(os.environ.get('RA_SHEET_N', '18'))]
    else:
        errs = [e for e in json.load(open(errors_p or ROOT / 'data' / 'door_v2' / 'learn_errors.json')) if e['day'] == day]
    ticks, spans = D._read(path)
    D.add_io(ticks, path)
    at = {round(r['t'], 2): r for r in ticks}
    tt = np.array(sorted(at))
    truth = D.truth(day, True)
    cam = D.run_cam(path)
    raw = D.Raw(day, cam)
    full = os.environ.get('RA_SHEET_FULL') == '1'
    out = ROOT / 'data' / 'door_v2' / 'sheets'
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for k, e in enumerate(sorted(errs, key=lambda e: e['t'])):
        tiles = []
        for off in OFFS:
            t = e['t'] + off
            j = int(np.argmin(np.abs(tt - t)))
            r = at[tt[j]]
            tk = SS.tick_frames(day, cam, r['t'], 1)[0]
            img = raw.get(tk)
            img = np.zeros((1224, 2176, 3), np.uint8) if img is None else cv2.resize(img, (2176, 1224), interpolation=cv2.INTER_AREA)
            for q in r['p']:
                cx, cy, w, h = q['box']
                x1, y1, x2, y2 = int((cx - w / 2) * 2176), int((cy - h / 2) * 1248), int((cx + w / 2) * 2176), int((cy + h / 2) * 1248)
                side = 'oid'[int(np.argmax(q.get('io_mix', q['io'])))]
                col = (0, 0, 255) if q['w'] == e.get('w') else (0, 255, 0)
                cv2.rectangle(img, (x1, y1), (x2, y2), col, 4)
                cv2.putText(img, '%s %d' % (side, q['w'] % 10000), (x1, max(30, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 1.6, col, 4)
                if q.get('foot'):
                    cv2.circle(img, (int(q['foot'][0]), int(q['foot'][1])), 10, col, -1)
            tile = cv2.resize(img if full else img[:900, :1500], (TW, TH), interpolation=cv2.INTER_AREA)   # the door corner, larger
            cv2.putText(tile, '%+.1fs' % off, (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            tiles.append(tile)
        near = ['%s%s %+.1f' % (x['kind'], '(staff)' if x['staff'] else '', x['t'] - e['t']) for x in truth if abs(x['t'] - e['t']) <= 10]
        bar = np.zeros((34, TW * len(OFFS), 3), np.uint8)
        txt = '#%d %s %s t=%.1f w=%s p=%s | owner: %s' % (k, e['what'].upper(), e['kind'], e['t'], e.get('w', '-') if e.get('w') is None else e['w'] % 10000,
                                                      e.get('p', '-'), ', '.join(near) or 'nothing')
        cv2.putText(bar, txt, (6, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        rows.append(np.vstack([bar, np.hstack(tiles)]))
    for p in range(0, len(rows), 6):
        cv2.imwrite(str(out / ('%s%s_%02d.jpg' % (day, os.environ.get('RA_SHEET_TAG', ''), p // 6))), np.vstack(rows[p:p + 6]), [cv2.IMWRITE_JPEG_QUALITY, 85])
    print(len(rows), 'errors drawn')


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else None)
