"""09.10: the role model learns from the owner's /liveevents answers (staff_live_learn.py) -- on a synthetic day where
the current model calls a worker in a new outfit a customer; the owner's answers teach it, and the swap happens only
because the refit is better on days it did not see."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'multicam'))


def unit(a):
    a = np.asarray(a, np.float32)
    return a / np.linalg.norm(a, axis=-1, keepdims=True)


@pytest.fixture
def world(tmp_path):
    import cv2
    rng = np.random.default_rng(0)
    D = 160                                          # PCA keeps 128 / 64 components
    # the base set: customers around one look, staff around another
    cust, staff = unit(rng.normal(size=D)), unit(rng.normal(size=D))
    n = 200
    y = np.arange(n) < 60
    cloth = unit(np.where(y[:, None], staff, cust) + 0.35 * rng.normal(size=(n, D)))
    shape = unit(rng.normal(size=(n, 80)))
    boxes = np.tile(np.array([[500, 200, 600, 600]], np.float32), (n, 1))
    np.savez(tmp_path / 'base.npz', cloth=cloth, shape=shape, y=y, boxes=boxes, cam=np.array(['cam1'] * n))
    # live days: a worker in a NEW outfit (unlike the base staff), the owner says 'staff'; customers as before
    new_staff = unit(rng.normal(size=D))
    live = tmp_path / 'live'
    vec = {}
    events = []
    for day, clock in (('20261009', '2026-10-09'), ('20261010', '2026-10-10'), ('20261011', '2026-10-11')):
        run = live / 'role_views' / (day + '_100000')
        run.mkdir(parents=True)
        views = {}
        for i in range(8):
            key = 'w%s:%d' % (day, i)
            is_staff = i < 4
            look = new_staff if is_staff else cust
            views[key] = []
            for j in range(4):
                name = '%d_%d.jpg' % (i, j)
                cv2.imwrite(str(run / name), np.full((64, 32, 3), 10 * i + j, np.uint8))
                vec[str(run / name)] = (unit(look + 0.2 * rng.normal(size=D)), unit(rng.normal(size=80)))
                views[key].append([1000.0 - j, name, [500, 200, 600, 600]])
            # the live door told a customer for everyone (the model does not know the new outfit)
            events.append({'kind': 'in', 't': 1000.0 * i, 'clock': '%s 10:%02d:00' % (clock, i), 'role': 'customer',
                           'tracks': [key], 'p_staff': 0.1})
        json.dump(views, open(run / 'views.json', 'w'))
    (live / 'review').mkdir()
    import live_events as LE
    for day in ('20261009', '20261010', '20261011'):
        st = {}
        for e in events:
            if e['clock'][:10].replace('-', '') == day:
                staff = int(e['tracks'][0].split(':')[1]) < 4
                st[LE.key_of(e)] = {'v': 4 if staff else 1}      # 'role wrong' on the workers, 'right' on customers
        json.dump(st, open(live / 'review' / ('%s.json' % day), 'w'))
    (live / 'events_cam1.jsonl').write_text('\n'.join(json.dumps(e) for e in events) + '\n')
    return tmp_path, live, vec


def test_owner_answers_teach_the_role(world, monkeypatch):
    import live_events as LE
    import staff_live_learn as SL
    tmp, live, vec = world
    monkeypatch.setattr(LE, 'LIVE', live)
    monkeypatch.setattr(LE, 'REVIEW', live / 'review')
    monkeypatch.setattr(SL, 'OUT', tmp / 'out')
    monkeypatch.setattr(SL, 'MIN_PEOPLE', 5)

    def fake_vectors(paths, embed=None):                # the ReID teachers are not on the Mac
        keep = [p for p in paths if p in vec]
        return keep, np.stack([vec[p][0] for p in keep]), np.stack([vec[p][1] for p in keep])
    monkeypatch.setattr(SL, 'vectors', fake_vectors)
    z = np.load(tmp / 'base.npz')
    current = SL.fit(z['cloth'], z['shape'], z['boxes'], z['cam'], z['y'])
    rep = SL.main(live=live, base=tmp / 'base.npz', current=current, out_root=tmp / 'out')
    assert rep['people'] == 24 and rep['views'] == 96
    assert rep['current_on_live']['staff_as_customer'] >= 15            # the new outfit: many worker views missed
    assert rep['refit_on_live_lodo']['staff_as_customer'] < rep['current_on_live']['staff_as_customer']
    assert rep['refit_on_live_lodo']['right'] > rep['current_on_live']['right']
    assert rep['decision'] == 'replaced'
    out = tmp / 'out' / sorted(p.name for p in (tmp / 'out').iterdir())[0]
    assert (out / 'report.json').exists() and (out / 'current_role.pkl').exists()


def test_too_few_answers_change_nothing(world, monkeypatch):
    import live_events as LE
    import staff_live_learn as SL
    tmp, live, vec = world
    monkeypatch.setattr(LE, 'LIVE', live)
    monkeypatch.setattr(LE, 'REVIEW', live / 'review')
    monkeypatch.setattr(SL, 'MIN_PEOPLE', 100)
    rep = SL.main(live=live, base=tmp / 'base.npz', current=object(), out_root=tmp / 'out')
    assert rep['people'] == 24 and rep['decision'].startswith('not enough') and not (tmp / 'out').exists()
