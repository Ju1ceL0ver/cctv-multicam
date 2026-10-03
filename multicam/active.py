"""Active choice for /keep2: three networks trained on the owner's answers so far, the people he has not answered scored by
all three, the ones the networks are least sure of (middle of the ignore band, the seeds disagreeing) put first in his queue.
usage: active.py [N=300]  -> data/keep2/priority.json {sample id: rank}"""
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import keepnet as K


def main(n=300):
    t0 = time.time()
    rows = K.load_keep2()
    new = K.load_keep2(unanswered=True)
    print('answered %d, to score %d, %.0f s' % (len(rows), len(new), time.time() - t0), flush=True)
    P = np.stack([K.score(K.fit(rows, 12, seed=s), new) for s in range(3)])
    m, sd = P.mean(0), P.std(0)
    band = (m >= 0.02) & (m <= 0.75)
    key = np.where(band, -sd - 0.3 * (1 - np.abs(m - 0.3)), 9 + np.abs(m - 0.3))     # in the band first, the seeds' disagreement decides
    order = np.argsort(key)[:n]
    pri = {new[i]['id']: r for r, i in enumerate(order)}
    json.dump(pri, open(K.ROOT / 'data' / 'keep2' / 'priority.json', 'w'))
    json.dump({new[i]['id']: [round(float(m[i]), 3), round(float(sd[i]), 3)] for i in range(len(new))}, open(K.ROOT / 'data' / 'keep2' / 'scores_all.json', 'w'))
    print('in the band %d of %d (%.0f%%), chosen %d, %.0f s' % (band.sum(), len(new), 100 * band.mean(), len(pri), time.time() - t0), flush=True)
    print('deleted by >=0.75: %d, kept <0.02: %d' % ((m > 0.75).sum(), (m < 0.02).sum()), flush=True)


if __name__ == '__main__':
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 300)
