"""The exam's frames (the owner's 18.09 painted ones) at full resolution from the raw recording, as the model
sees training frames: 2176 x 1224 -> data/v2_exam/<id>.jpg. Each is checked against the painted 1280 x 720
frame (mean absolute difference after shrinking); a frame that does not match is not written."""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
RAW = ROOT.parent / 'retail_analytics' / 'runs' / 'raw'
OUT = ROOT / 'data' / 'v2_exam'


def main():
    import gold
    where = json.load(open(ROOT / 'data' / 'seg_datasets' / 'backgrounds' / 'exam_frames.json'))
    OUT.mkdir(parents=True, exist_ok=True)
    report = {}
    for ident, w in sorted(where.items()):
        painted = cv2.imread(str(gold.PAINT / ('%s.jpg' % ident)))
        cap = cv2.VideoCapture(str(RAW / w['cam'] / w['day'] / w['segment']))
        best = None
        for dt in (0.0, -0.04, 0.04, -0.08, 0.08):                # the painted frame is one of the nearest few
            cap.set(cv2.CAP_PROP_POS_MSEC, (w['second'] + dt) * 1000)
            ok, img = cap.read()
            if not ok:
                continue
            d = float(np.abs(cv2.resize(img, (1280, 720), interpolation=cv2.INTER_AREA).astype(np.float32) - painted.astype(np.float32)).mean())
            if best is None or d < best[0]:
                best = (d, img)
        cap.release()
        if best is None or best[0] > 6.0:
            report[ident] = {'ok': False, 'diff': None if best is None else round(best[0], 2)}
            continue
        cv2.imwrite(str(OUT / ('%s.jpg' % ident)), cv2.resize(best[1], (2176, 1224), interpolation=cv2.INTER_AREA), [cv2.IMWRITE_JPEG_QUALITY, 95])
        report[ident] = {'ok': True, 'diff': round(best[0], 2)}
    json.dump(report, open(OUT / 'report.json', 'w'), indent=1)
    print(sum(r['ok'] for r in report.values()), 'of', len(report), 'frames; worst diff', max((r['diff'] or 0) for r in report.values()))


if __name__ == '__main__':
    main()
