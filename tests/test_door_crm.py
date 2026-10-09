"""09.10: the live door's crossings -> the Refloor CRM cards (door_crm.py): each card once, failures retried, the role
waited for, nothing from before `since`."""
import json
import sys
import time
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'multicam'))
import door_crm  # noqa: E402


class Card:
    def __init__(self, fields):
        self.fields, self.record = fields, None


def card_from_event(ev, cfg):
    role = cfg.role_map.get(ev.role, 'guest')
    return Card({'tt': cfg.tt, 'camera': cfg.camera_code, 'role': role, 'time': ev.moment.strftime('%Y-%m-%d %H:%M:%S'),
                 'probability': '%.4f' % (ev.staff_probability if role == 'employee' else 1 - ev.staff_probability),
                 'event': {'entry': 'detect', 'exit': 'left'}[ev.event]})


class Sink:
    url = 'https://crm.example/add/'
    last_error = ''

    def __init__(self, fail=()):
        self.posted, self.fail = [], set(fail)

    def _post(self, card):
        self.posted.append(dict(card.fields))
        if card.fields['time'] in self.fail:
            self.last_error = 'HTTP 503'
            return False
        return True


def setup(tmp, monkeypatch, sink):
    live = tmp / 'live'
    (live / 'records').mkdir(parents=True)
    mod = types.ModuleType('retail_analytics.sinks.crm_sink')
    mod.card_from_event = card_from_event
    for name in ('retail_analytics', 'retail_analytics.sinks'):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    monkeypatch.setitem(sys.modules, 'retail_analytics.sinks.crm_sink', mod)
    cfg = types.SimpleNamespace(tt='NSK-01', camera_code='cam-1', role_map={'staff': 'employee', 'customer': 'guest'})
    monkeypatch.setattr(door_crm, 'LIVE', live)
    monkeypatch.setattr(door_crm, 'OUT', live / 'crm')
    monkeypatch.setattr(door_crm, 'sink', lambda dry: (sink, cfg))
    return live


def event(kind, t, role=None):
    return {'kind': kind, 't': t, 'role': role, 'p_staff': None if role is None else (0.9 if role == 'staff' else 0.1),
            'tracks': ['w:1'], 'clock': time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(t)), 'photo': '', 'crop': ''}


def write(live, evs, roles=()):
    (live / 'events_cam1.jsonl').write_text(''.join(json.dumps(e) + '\n' for e in evs), encoding='utf-8')
    day = time.strftime('%Y%m%d', time.localtime(evs[0]['t']))
    (live / 'records' / ('roles_%s.jsonl' % day)).write_text(''.join(json.dumps(r) + '\n' for r in roles), encoding='utf-8')


def test_each_card_once_old_events_left_out_role_waited(tmp_path, monkeypatch):
    import live_events as LE
    s = Sink()
    live = setup(tmp_path, monkeypatch, s)
    t0 = 1791500000.0
    old, a, b = event('in', t0 - 100, 'customer'), event('in', t0 + 1), event('out', t0 + 5, 'staff')
    write(live, [old, a, b], [{'key': LE.key_of(a), 'role': 'staff', 'p_staff': 0.8}])
    st = door_crm.main(since=t0, once=True)
    assert [p['event'] for p in s.posted] == ['detect', 'left']            # the old event is not sent
    assert s.posted[0]['role'] == 'employee' and s.posted[0]['probability'] == '0.8000'   # the role model's answer
    assert s.posted[1]['role'] == 'employee'
    assert all(c['status'] == 'sent' for c in st['cards'].values())
    door_crm.main(once=True)                                               # a restart: nothing goes twice
    assert len(s.posted) == 2


def test_role_waited_then_default(tmp_path, monkeypatch):
    s = Sink()
    live = setup(tmp_path, monkeypatch, s)
    t0 = 1791500000.0
    write(live, [event('in', t0 + 1)])
    door_crm.main(since=t0, once=True)
    assert s.posted == []                                                  # no role yet: waits
    monkeypatch.setattr(door_crm, 'ROLE_WAIT', 0.0)
    door_crm.main(once=True)
    assert [p['role'] for p in s.posted] == ['guest']                      # waited enough: as a customer


def test_failure_retried_until_it_goes(tmp_path, monkeypatch):
    t0 = 1791500000.0
    e = event('out', t0 + 1, 'customer')
    s = Sink(fail={time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(e['t']))})
    live = setup(tmp_path, monkeypatch, s)
    write(live, [e])
    st = door_crm.main(since=t0, once=True)
    assert list(st['cards'].values())[0]['status'] == 'retry'
    door_crm.main(once=True)
    assert len(s.posted) == 1                                              # not before RETRY seconds
    monkeypatch.setattr(door_crm, 'RETRY', 0.0)
    s.fail.clear()
    st = door_crm.main(once=True)
    assert len(s.posted) == 2 and list(st['cards'].values())[0]['status'] == 'sent'
    assert list(st['cards'].values())[0]['tries'] == 2


def test_snapshot_boxes_the_person(tmp_path, monkeypatch):
    import numpy as np
    import cv2
    import door_live
    monkeypatch.setattr(door_live, 'LIVE', tmp_path)
    fr = np.zeros((720, 1280, 3), np.uint8)
    fr[300:420, 600:680] = 200                                      # the person, RGB as Window.frame gives it
    win = types.SimpleNamespace(id='w1', times=[100.0, 100.24, 100.48], frame=lambda k: fr)
    k = 1224 / 1248                                                 # people() boxes: y by H * 1248 / 1224
    ticks = [{'t': 100.24, 'p': [{'w': 1005, 'box': [640 / 1280, 360 / 720 * k, 80 / 1280, 120 / 720 * k]},
                                 {'w': 1006, 'box': [0.1, 0.1, 0.05, 0.05]}]}]
    full, crop = door_live.snapshot(win, ticks, {'kind': 'in', 't': 100.3, 'w': 'w1:1005', 'tracks': ['w1:1005']})
    c = cv2.imread(crop)
    assert c is not None and abs(c.shape[1] - (80 + 2 * (12 + 4))) <= 2 and abs(c.shape[0] - (120 + 2 * (9 + 4))) <= 2
    assert c[c.shape[0] // 2, c.shape[1] // 2].mean() > 150                     # the person is in the middle of the crop
    f = cv2.imread(full)
    assert f.shape == (720, 1280, 3) and f[300, 640, 1] > 200                   # the green box on the full frame
    assert door_live.snapshot(win, ticks, {'kind': 'in', 't': 100.3, 'w': 'w1:9', 'tracks': ['w1:9']}) == ('', '')
