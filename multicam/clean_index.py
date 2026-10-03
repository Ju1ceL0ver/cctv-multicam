"""Frames of data/sam31_stills_clean that still hold an ignored person (value 254) -> _ignore_frames.json (v2_data leaves them out)."""
import json
import cv2
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import sys
D = Path(__file__).resolve().parent / 'data' / (sys.argv[1] if len(sys.argv) > 1 else 'sam31_stills_clean')


def has(p):
    m = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
    return p.stem if m is not None and (m == 254).any() else None


if __name__ == '__main__':
    fs = sorted(D.glob('*.png'))
    with ThreadPoolExecutor(8) as ex:
        bad = [x for x in ex.map(has, fs) if x]
    json.dump(bad, open(D / '_ignore_frames.json', 'w'))
    print('frames %d, with an ignored person %d, usable %d' % (len(fs), len(bad), len(fs) - len(bad)))
