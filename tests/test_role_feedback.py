"""09.10: the owner's /liveevents answers feed the live door's staff gallery (door_role.Roles.feedback)."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'multicam'))
import door_role  # noqa: E402


class FakeBank:
    def __init__(self, views):
        self.views = views                                     # key -> [(score, path, box)]

    def best(self, keys, n=8):
        return sorted((x for k in keys for x in self.views.get(k, [])), key=lambda z: -z[0])[:n]


class FakeWorker:
    threshold = 0.45

    def __init__(self, p, vec):
        self.p, self.v = p, vec

    def probs(self, paths, boxes, vectors=False):
        return [self.p[x] for x in paths], [self.v[x] for x in paths]


def key_of(e):
    return '%s_%.1f' % (e['kind'], e['t'])


def unit(a):
    a = np.asarray(a, np.float32)
    return a / np.linalg.norm(a)


def make(n_staff_views=6):
    rng = np.random.default_rng(0)
    staff_look = unit(rng.normal(size=16))
    views, p, vec = {}, {}, {}
    for who, look, ps in (('a:1', staff_look, 0.2), ('b:1', staff_look, 0.2), ('c:1', unit(rng.normal(size=16)), 0.1)):
        views[who] = []
        for i in range(n_staff_views):
            path = '%s_%d.jpg' % (who, i)
            views[who].append((1.0 - i * 0.01, path, [0, 0, 1, 1]))
            p[path] = ps
            vec[path] = unit(look + 0.05 * rng.normal(size=16))
    return door_role.Roles(FakeBank(views), FakeWorker(p, vec))


def test_owner_marks_a_customer_as_staff_and_the_same_clothes_follow():
    r = make()
    # the model calls everyone a customer (p 0.2 < 0.45) and there is no gallery yet
    ev = {'kind': 'in', 't': 1.0, 'tracks': ['a:1'], 'role': r.role(['a:1'])['role']}
    assert ev['role'] == 'customer'
    assert r.role(['b:1'])['role'] == 'customer'
    took = r.feedback([ev], {key_of(ev): {'v': 4}}, key_of)     # the owner: the role is wrong -> staff
    assert took == 1
    assert r.role(['a:1']) == dict(r.role(['a:1']), role='staff', by='owner')
    # another track in the same clothes: staff by the gallery the owner's answer seeded (6 views >= GALLERY_MIN)
    out = r.role(['b:1'])
    assert out['role'] == 'staff' and out['gallery_sim'] > 0.9
    assert r.role(['c:1'])['role'] == 'customer'
    assert r.feedback([ev], {key_of(ev): {'v': 4}}, key_of) == 0     # the same answer is taken once


def test_owner_marks_staff_as_customer_and_its_views_leave_the_gallery():
    r = make()
    ev = {'kind': 'out', 't': 2.0, 'tracks': ['a:1'], 'role': 'customer'}
    r.role(['a:1'])
    r.feedback([ev], {key_of(ev): {'v': 4}}, key_of)               # -> staff, seeds the gallery
    assert r.role(['b:1'])['role'] == 'staff'
    ev2 = {'kind': 'in', 't': 3.0, 'tracks': ['a:1'], 'role': 'staff'}
    r.feedback([ev2], {key_of(ev2): {'v': 4}}, key_of)             # the owner changes their mind: a customer
    assert r.role(['a:1'])['role'] == 'customer'
    assert all(o != ('a:1',) for _, o in r.gallery)
    assert r.role(['b:1'])['role'] == 'customer'


def test_right_on_a_staff_event_confirms_and_other_answers_are_ignored():
    r = make()
    ev = {'kind': 'in', 't': 4.0, 'tracks': ['a:1'], 'role': 'staff'}
    r.role(['a:1'])
    assert r.feedback([ev], {key_of(ev): {'v': 1}}, key_of) == 1   # right -> the staff role confirmed
    assert r.role(['a:1'])['role'] == 'staff'
    ev3 = {'kind': 'in', 't': 5.0, 'tracks': ['c:1'], 'role': 'customer'}
    assert r.feedback([ev3], {key_of(ev3): {'v': 2}}, key_of) == 0  # 'nobody crossed' says nothing of the role
    assert r.role(['c:1'])['role'] == 'customer'


def test_rescore_changes_an_early_role_when_the_gallery_grows():
    """09.10: the morning event of a worker the model missed becomes staff once the gallery has her clothes"""
    import time
    r = make()
    today = time.strftime('%Y-%m-%d') + ' 10:00:00'
    early = {'kind': 'in', 't': 1.0, 'clock': today, 'tracks': ['b:1']}
    early.update(r.role(['b:1']))
    assert early['role'] == 'customer'
    # later the model becomes sure of someone in the same clothes (p >= GALLERY_SEED with >= MIN_SEED_VIEWS views)
    for _, path, _ in r.bank.views['a:1']:
        r.worker.p[path] = 0.9
    r.cache = {}
    assert r.role(['a:1'])['role'] == 'staff'
    ch = r.rescore([early])
    assert len(ch) == 1 and ch[0][1] == 'customer' and early['role'] == 'staff' and early['gallery_sim'] > 0.9
    assert r.rescore([early]) == []                                  # nothing new the second time
    old = {'kind': 'in', 't': 2.0, 'clock': '2000-01-01 10:00:00', 'tracks': ['c:1'], 'role': 'staff'}
    assert r.rescore([old]) == [] and old['role'] == 'staff'          # another day's events are left alone
