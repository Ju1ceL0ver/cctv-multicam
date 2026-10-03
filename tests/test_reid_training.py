import os
import random
import sys
from datetime import datetime

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'multicam'))


def test_track_is_cut_where_the_look_changes_for_good():
    import reid_harvest
    rng = np.random.default_rng(0)
    a, b = rng.normal(size=512), rng.normal(size=512)
    looks = np.r_[a + 0.05 * rng.normal(size=(40, 512)), b + 0.05 * rng.normal(size=(40, 512))]
    cuts = reid_harvest.jumps(looks)
    assert len(cuts) == 1 and abs(cuts[0] - 40) <= 1
    assert reid_harvest.jumps(a + 0.05 * rng.normal(size=(80, 512))) == []


def test_only_big_sure_untouched_looks_are_kept():
    import reid_harvest
    #            t    x1   y1   x2   y2   score
    dets = np.array([[0.0, 100, 100, 200, 500, 0.9, 0, 0, 0, 0],      # alone, big: kept
                     [0.1, 100, 100, 200, 500, 0.9, 0, 0, 0, 0],      # touched by the next one
                     [0.1, 190, 100, 290, 500, 0.9, 0, 0, 0, 0],
                     [0.2, 100, 100, 200, 200, 0.9, 0, 0, 0, 0],      # too small
                     [0.3, 100, 100, 200, 500, 0.3, 0, 0, 0, 0]], np.float32)   # detector unsure
    assert reid_harvest.clean_rows(dets, [0, 1, 3, 4]).tolist() == [0]


def test_only_simultaneous_pieces_are_rivals():
    import train_reid
    people = [dict(day='d', cam='cam1', first=0, last=10), dict(day='d', cam='cam1', first=5, last=20),
              dict(day='d', cam='cam1', first=30, last=40),                   # later: may be the same person
              dict(day='d', cam='cam2', first=0, last=10)]                    # other camera: may be the same person
    riv = train_reid.rivals(people)
    assert riv == [[1], [0], [], []]
    groups = next(train_reid.batches(people, riv, random.Random(0)))
    assert all(set(g) <= {0, 1} for g in groups)


def test_night_job_waits_for_the_evening():
    import night_reid
    assert night_reid.next_start(datetime(2026, 9, 23, 8, 40)) == datetime(2026, 9, 23, 21, 2)
    assert night_reid.next_start(datetime(2026, 9, 23, 22, 26)) == datetime(2026, 9, 23, 22, 26)
    assert night_reid.next_start(datetime(2026, 9, 24, 3, 0)) == datetime(2026, 9, 24, 3, 0)
    assert night_reid.next_start(datetime(2026, 9, 24, 9, 30)) == datetime(2026, 9, 24, 21, 2)
