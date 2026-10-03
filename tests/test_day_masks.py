"""Click a person, get his outline: exact frames, click-to-mask, drawn people carried through
the film and joined to the track that picks them up."""
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'multicam'))
import day_masks as dmk                              # noqa: E402
import day_movie as dm                               # noqa: E402
import day_player as dp                              # noqa: E402
from storage import atomic_json                      # noqa: E402

DAY = '20260917'
T0 = datetime(2026, 9, 17, 10, 0, 0)
FFMPEG = shutil.which('ffmpeg')
W, H = 960, 540
GREEN = (40, 200, 40)
pytestmark = pytest.mark.skipif(not FFMPEG, reason='needs ffmpeg')


def person_at(t):
    """The one walker of this day, in copy pixels: from x=100 to x=500 over 20 s."""
    x = 100 + 20 * t
    return [x, 200, x + 40, 360]


def frame(k):
    image = np.full((H, W, 3), 110, np.uint8)
    x1, y1, x2, y2 = [int(round(v)) for v in person_at(k * 0.04)]
    image[y1:y2, x1:x2] = GREEN
    for b in range(10):                              # the frame number, for checking exactness
        image[500:530, 10 + b * 40:40 + b * 40] = 255 if (k >> b) & 1 else 0
    return image


def number(image):
    return sum(1 << b for b in range(10) if image[510:520, 20 + b * 40:30 + b * 40].mean() > 128)


class Green:
    """A segmenter that knows the walker is green: the carrying logic, not the model, is tested."""
    name = 'green'

    def __init__(self):
        self.image = {}

    def encode(self, slot, key, image):
        self.image[slot] = image
        return True

    def mask(self, slot, points, labels, box=None):
        image = self.image[slot]
        return (image[:, :, 1] > 180) & (image[:, :, 0] < 80)


@pytest.fixture
def day(tmp_path, monkeypatch):
    root = tmp_path / 'p'
    for cam in ('cam1', 'cam2'):
        folder = root / 'data/day_proxy' / DAY / cam
        folder.mkdir(parents=True)
        pipe = subprocess.Popen([FFMPEG, '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-s', '%dx%d' % (W, H),
                                 '-r', '25', '-i', '-', '-c:v', 'libx264', '-crf', '14', '-g', '100', '-preset', 'ultrafast',
                                 '-pix_fmt', 'yuv420p', str(folder / '100000_0000.mp4')], stdin=subprocess.PIPE)
        for k in range(500):
            pipe.stdin.write(frame(k).tobytes())
        pipe.stdin.close()
        assert pipe.wait() == 0
        atomic_json(folder / '100000_0000.json', {'day': DAY, 'cam': cam, 'raw': '100000_0000.mp4', 'start': T0.isoformat(),
                                                  'frames': 500, 'breaks': [[0, 0.0]], 'shift': 0.0})
    atomic_json(root / 'data/cam_sync.json', {DAY: {'cam1_to_cam2_s': 0.0}})
    # the tracker picks the walker up at 12 s and holds him to 16 s
    clip = root / 'data/raw_clips/c100000'
    clip.mkdir(parents=True)
    rows = []
    for k in range(300, 400, 3):
        row = np.zeros(10, np.float32)
        row[0] = k / 25.0
        row[1:5] = np.array(person_at(k * 0.04)) * 2560 / W
        row[5] = 0.9
        rows.append(row)
    atomic_json(clip / 'meta_yolo26x-seg.json', {'day': DAY, 'start': T0.isoformat(), 'seconds': 20})
    atomic_json(clip / 'pieces_yolo26x-seg.json', [{'piece': 0, 'cam': 'cam1', 't0': 12.0, 't1': 16.0,
                                                   'dets': list(range(len(rows)))}])
    np.savez(clip / 'dets_yolo26x-seg.npz', cam1=np.array(rows, np.float32), cam2=np.zeros((0, 10), np.float32))
    monkeypatch.setattr(dp, '_raw_segments', lambda cam, day: [])
    dp._cache.clear(); dm._cache.clear(); dm._clip_cache.clear()
    dmk.FRAMES.open.clear()
    return root


def test_the_frame_for_a_moment_is_the_one_recorded_then(day):
    for t, k in ((7.0, 175), (7.03, 175), (7.04, 176), (3.5, 87), (19.96, 499)):
        image, info = dmk.FRAMES.at(DAY, 'cam1', t, day)
        assert info['index'] == k and number(image) == k, (t, info)
    image, _ = dmk.FRAMES.at(DAY, 'cam1', 2.0, day)          # going back reopens and still lands exactly
    assert number(image) == 50
    assert dmk.FRAMES.at(DAY, 'cam1', 25.0, day) == (None, None)


def test_a_click_on_a_person_gives_his_outline(day, monkeypatch):
    monkeypatch.setattr(dmk, 'SEGMENTER', dmk.Segmenter('grabcut'))
    x1, y1, x2, y2 = person_at(4.0)
    k = 2560 / W
    answer = dmk.segment(DAY, 'cam1', 4.0, [[(x1 + x2) / 2 * k, (y1 + y2) / 2 * k, 1]], root=day)
    assert answer['frame']['index'] == 100 and answer['rings']
    bx = answer['box']
    truth = [x1 * k, y1 * k, x2 * k, y2 * k]
    assert dmk._iou(bx, truth) > 0.6, (bx, truth)


def test_a_drawn_person_is_carried_until_the_tracker_has_him_and_then_joined(day):
    k = 2560 / W
    x1, y1, x2, y2 = [v * k for v in person_at(4.0)]
    ring = [int(x1), int(y1), int(x2), int(y1), int(x2), int(y2), int(x1), int(y2)]
    made = dmk.draw(DAY, {'cam': 'cam1', 'at': 4.0, 'rings': [ring]}, day)
    dmk.carry(DAY, made['id'], day, segmenter_=Green())
    item = dmk.drawn(DAY, day)['items'][str(made['id'])]
    times = sorted(s[0] for s in item['samples'])
    assert item['carry']['state'] == 'done' and item['joins'] == ['c100000:0@']
    assert times[0] < 0.3 and 11.5 <= times[-1] < 12.0          # back to the start, forward to the tracker
    assert max(np.diff(times)) <= dmk.STEP + 1e-6
    crowd = dm.people(DAY, day)
    drawn_person = crowd['parts'][made['track'] + '@']['person']
    assert drawn_person == crowd['parts']['c100000:0@']['person']       # one person
    view = dm.overview(DAY, day)
    assert view['drawn'][0]['track'] == 'd1' and len(view['drawn'][0]['samples']) == len(times)


def test_a_drawn_person_who_is_nobody_known_is_his_own_person_and_can_be_named(day):
    ring = [2200, 100, 2300, 100, 2300, 400, 2200, 400]          # somebody the tracker never had
    dmk.draw(DAY, {'cam': 'cam1', 'at': 1.0, 'rings': [ring]}, day)
    crowd = dm.people(DAY, day)
    machine = crowd['parts']['c100000:0@']['person']
    drawn_person = crowd['parts']['d1@']['person']
    # one frame, off the shop floor: the same rule as for anybody -- a passer-by, no number
    assert drawn_person != machine and crowd['persons'][drawn_person]['kind'] == 'passer'
    assert crowd['persons'][drawn_person]['n'] is None
    dm.change(DAY, {'action': 'kind', 'revision': 0, 'part': 'd1@', 'kind': 'customer'}, day)
    crowd = dm.people(DAY, day)
    assert crowd['persons'][drawn_person]['kind'] == 'customer' and crowd['persons'][drawn_person]['n'] == 1
    dmk.undraw(DAY, 1, day)
    assert 'd1@' not in dm.people(DAY, day)['parts']


def test_a_fix_replaces_the_outline_without_making_a_person(day):
    made = dmk.draw(DAY, {'cam': 'cam1', 'at': 12.0, 'rings': [[100, 100, 200, 100, 200, 300]],
                          'replaces': 'c100000:0'}, day)
    assert dmk.drawn(DAY, day)['items'][str(made['id'])]['carry'] is None
    assert not any(p.startswith('d') for p in dm.people(DAY, day)['parts'])
    fix = dm.overview(DAY, day)['drawn'][0]
    assert fix['kind'] == 'fix' and fix['replaces'] == 'c100000:0'
    with pytest.raises(ValueError, match='Нечего'):
        dmk.draw(DAY, {'cam': 'cam1', 'at': 1.0, 'rings': []}, day)


def test_routes(day, monkeypatch):
    from flask import Flask
    monkeypatch.setattr(dmk, 'SEGMENTER', dmk.Segmenter('grabcut'))
    monkeypatch.setattr(dmk, 'carry_later', lambda *a, **k: None)
    app = Flask(__name__)
    dmk.register(app, lambda: day)
    client = app.test_client()
    k = 2560 / W
    x1, y1, x2, y2 = person_at(6.0)
    got = client.post('/api/movie/%s/segment' % DAY, json={'cam': 'cam1', 'at': 6.0,
                                                          'points': [[(x1 + x2) / 2 * k, (y1 + y2) / 2 * k, 1]]}).get_json()
    assert got['rings'] and got['frame']['index'] == 150
    assert client.post('/api/movie/%s/segment' % DAY, json={'cam': 'cam1', 'at': 6.0, 'points': []}).status_code == 400
    assert client.post('/api/movie/%s/warm' % DAY, json={'at': 6.0}).get_json()['queued']
    made = client.post('/api/movie/%s/draw' % DAY, json={'cam': 'cam1', 'at': 6.0, 'rings': got['rings']}).get_json()
    assert made['track'] == 'd1'
    assert client.post('/api/movie/%s/undraw' % DAY, json={'id': 1}).get_json()['deleted'] is True
