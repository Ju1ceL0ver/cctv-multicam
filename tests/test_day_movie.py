"""The day as a film: both cameras on one clock, the machine's people, the owner's corrections."""
import gzip
import json
import shutil
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'multicam'))
import day_movie as dm                               # noqa: E402
import day_player as dp                              # noqa: E402
from storage import atomic_json, read_json           # noqa: E402

DAY = '20260917'
T0 = datetime(2026, 9, 17, 10, 0, 0)
FFMPEG = shutil.which('ffmpeg')
BITS = 10


def _pattern(n):
    """Frame number written into the picture as ten big black/white squares."""
    image = np.full((360, 640, 3), 90, np.uint8)
    for b in range(BITS):
        if (n >> b) & 1:
            image[100:160, 20 + b * 60:70 + b * 60] = 255
        else:
            image[100:160, 20 + b * 60:70 + b * 60] = 0
    return image


def _read_pattern(image):
    if image[20:80].mean() < 50:                         # the dark frame of a hole (the pattern sits on grey 90)
        return None
    return sum(1 << b for b in range(BITS) if image[125:135, 40 + b * 60:50 + b * 60].mean() > 128)


def _proxy(root, cam, start, frames, breaks):
    folder = root / 'data/day_proxy' / DAY / cam
    folder.mkdir(parents=True, exist_ok=True)
    name = start.strftime('%H%M%S') + '_0000'
    pipe = subprocess.Popen([FFMPEG, '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-s', '640x360',
                             '-r', '25', '-i', '-', '-c:v', 'libx264', '-crf', '12', '-preset', 'ultrafast',
                             '-pix_fmt', 'yuv420p', str(folder / (name + '.mp4'))], stdin=subprocess.PIPE)
    for k in range(frames):
        pipe.stdin.write(_pattern(k + 1).tobytes())     # +1: zero is the hole
    pipe.stdin.close()
    assert pipe.wait() == 0
    atomic_json(folder / (name + '.json'), {'day': DAY, 'cam': cam, 'raw': name + '.mp4', 'start': start.isoformat(),
                                            'frames': frames, 'breaks': breaks})


@pytest.fixture
def short_segments(monkeypatch):
    monkeypatch.setattr(dp, 'SEGMENT', 30)
    monkeypatch.setattr(dp, '_raw_segments', lambda cam, day: [])
    dp._cache.clear()
    dm._cache.clear()
    dm._clip_cache.clear()


@pytest.mark.skipif(not FFMPEG, reason='needs ffmpeg')
def test_every_frame_of_the_film_is_the_frame_the_camera_had_shot_by_then(tmp_path, short_segments):
    """Camera 1 lost two seconds in one burst; camera 2 started later and sees the shop
    4 s late. Tick m of both films must show the last frame shot by 0.08*m on the shop
    clock -- frozen through the burst, dark where nothing was recorded."""
    root = tmp_path / 'p'
    _proxy(root, 'cam1', T0, 500, [[0, 0.0], [250, 12.0]])
    _proxy(root, 'cam2', T0 + timedelta(seconds=8), 300, [[0, 0.0]])
    atomic_json(root / 'data/cam_sync.json', {DAY: {'cam1_to_cam2_s': 4.0}})
    start, end = dm.span(DAY, root)
    assert (start, end) == (T0.timestamp(), T0.timestamp() + 34)
    def want(cam, s):
        if cam == 'cam1':
            if s > 21.96 + 0.5:
                return None
            if s < 12.0:
                return min(249, int(s / 0.04 + 1e-6))
            return min(499, 250 + int((s - 12.0) / 0.04 + 1e-6))
        if s < 4.0 - 1e-6 or s > 4.0 + 11.96 + 0.5:
            return None
        return min(299, int((s - 4.0) / 0.04 + 1e-6))
    import cv2
    for cam in ('cam1', 'cam2'):
        record = dm.build(DAY, cam, root, ffmpeg=FFMPEG, log=lambda s: None)
        assert record['frames'] == 425 and record['start'] == T0.timestamp()
        capture = cv2.VideoCapture(str(dm.film(DAY, cam, root)[0]))
        seen = []
        while True:
            ok, image = capture.read()
            if not ok:
                break
            seen.append(_read_pattern(image))
        assert len(seen) == 425
        wrong = []
        for m, got in enumerate(seen):
            s = round(m * 0.08, 4)
            want_ = want(cam, s)
            got = None if got is None else got - 1
            if got != want_:
                wrong.append((m, s, got, want_))
        assert not wrong, wrong[:8]
        if cam == 'cam2':
            assert record['holes'][0] == [0.0, 4.0] and record['holes'][-1][1] == 34.0


# ---------------------------------------------------------------- people

def _segments(root):
    for cam in ('cam1', 'cam2'):
        folder = root / 'data/day_proxy' / DAY / cam
        folder.mkdir(parents=True, exist_ok=True)
        for n in range(3):
            begin = T0 + timedelta(seconds=900 * n)
            name = '%s_%04d' % (T0.strftime('%H%M%S'), n)
            atomic_json(folder / (name + '.json'), {'day': DAY, 'cam': cam, 'raw': name + '.mp4',
                                                    'start': begin.isoformat(), 'frames': 22500, 'breaks': [[0, 0.0]]})


def _clip(root, name, start, tracks, groups=None, looks=None):
    """tracks: [(cam, [seconds...], box)]; seconds are clip-relative, as the pipeline stores them."""
    d = root / 'data/raw_clips' / name
    d.mkdir(parents=True)
    rows = {'cam1': [], 'cam2': []}
    faces = {'cam1': [], 'cam2': []}
    polys = {'cam1': [], 'cam2': []}
    pieces = []
    for index, (cam, moments, where) in enumerate(tracks):
        first = len(rows[cam])
        for n, t in enumerate(moments):
            row = np.zeros(10, np.float32)
            row[0] = t
            row[1:5] = where
            rows[cam].append(row)
            face = np.zeros(512, np.float32)
            face[(looks[index](n) if looks and index in looks else index) % 512] = 1.0
            faces[cam].append(face)
            x1, y1, x2, y2 = where
            polys[cam].append(np.array([[x1, y1], [x2, y1], [x2, y2], [(x1 + x2) / 2, y2 + 10], [x1, y2]], np.float32))
        pieces.append({'piece': index, 'cam': cam, 't0': float(moments[0]), 't1': float(moments[-1]),
                       'dets': list(range(first, len(rows[cam])))})
    atomic_json(d / 'meta_yolo26x-seg.json', {'day': DAY, 'start': start.isoformat(), 'seconds': 600})
    atomic_json(d / 'pieces_yolo26x-seg.json', pieces)
    atomic_json(d / 'groups_yolo26x-seg.json', [{'pieces': g} for g in (groups or [])])
    np.savez(d / 'dets_yolo26x-seg.npz',
             **{c: (np.array(v, np.float32) if v else np.zeros((0, 10), np.float32)) for c, v in rows.items()})
    np.savez(d / dm.EMB, **{c: (np.array(v, np.float32) if v else np.zeros((0, 512), np.float32)) for c, v in faces.items()})
    out = {}
    for c in ('cam1', 'cam2'):
        off = np.zeros(len(polys[c]) + 1, np.int64)
        for i, p in enumerate(polys[c]):
            off[i + 1] = off[i] + len(p)
        out[c + '_pts'] = np.concatenate(polys[c]) if polys[c] else np.zeros((0, 2), np.float32)
        out[c + '_off'] = off
    np.savez_compressed(d / 'polys_yolo26x-seg.npz', **out)
    return d


MID = [800, 300, 1000, 900]           # a person in the middle of the room
OTHER = [1400, 300, 1600, 900]


@pytest.fixture
def shop(tmp_path, monkeypatch):
    """Grid windows at 10:00:30 and 10:10:30 (the recording starts at 10:00:00). A walks
    through the seam between them on camera 1, and camera 2 sees A too. B stops in the
    middle of the room. An early clip at 10:05 lies under both windows and must not add a
    second outline to anybody."""
    monkeypatch.setattr(dp, '_raw_segments', lambda cam, day: [])
    dp._cache.clear(); dm._cache.clear(); dm._clip_cache.clear()
    root = tmp_path / 'p'
    _segments(root)
    atomic_json(root / 'data/cam_sync.json', {DAY: {'cam1_to_cam2_s': 0.0}})
    steps = list(np.arange(500, 600, 0.12))
    _clip(root, 'c20260917_100030', T0 + timedelta(seconds=30),
          [('cam1', steps, MID), ('cam2', steps, OTHER), ('cam1', list(np.arange(100, 160, 0.12)), OTHER)],
          groups=[[0, 1], [2]])
    _clip(root, 'c20260917_101030', T0 + timedelta(seconds=630),
          [('cam1', list(np.arange(0, 60, 0.12)), MID), ('cam2', list(np.arange(0, 60, 0.12)), OTHER)], groups=[[0, 1]])
    _clip(root, 'c100500', T0 + timedelta(seconds=300), [('cam1', list(np.arange(250, 300, 0.12)), MID)], groups=[[0]])
    return root


def test_the_machine_carries_a_person_across_the_seam_and_onto_camera_two(shop):
    crowd = dm.people(DAY, shop)
    person = {pid: p['person'] for pid, p in crowd['parts'].items()}
    a = person['c20260917_100030:0@']
    assert person['c20260917_101030:0@'] == a          # the seam: same spot, 0.12 s apart
    assert person['c20260917_100030:1@'] == a          # the grouping put camera 2 with it
    assert person['c20260917_100030:2@'] != a
    assert crowd['persons'][a]['n'] == 2 and crowd['persons'][person['c20260917_100030:2@']]['n'] == 1


def test_an_early_clip_under_the_grid_draws_nothing(shop):
    tracks = dm.index(DAY, shop)['tracks']
    assert not any(k.startswith('c100500:') for k in tracks)
    assert json.loads(gzip.decompress(dm.clip_marks(DAY, 'c100500', shop)))['tracks'] == []


def test_the_page_gets_outlines_on_the_film_clock_in_half_resolution(shop):
    body = json.loads(gzip.decompress(dm.clip_marks(DAY, 'c20260917_101030', shop)))
    assert body['unit'] == 1280
    track = next(t for t in body['tracks'] if t['k'] == 'c20260917_101030:0')
    assert track['cam'] == 1 and track['t'][0] == 630.0    # clip start 10:10:30 is 630 s into the film
    ring = np.array(track['p'][0][0]).reshape(-1, 2)        # first detection, first ring
    assert len(track['p'][0]) == 1
    assert abs(ring[:, 0].min() - 400) <= 2 and abs(ring[:, 0].max() - 500) <= 2      # 800..1000 at 2560
    assert abs(ring[:, 1].min() - 150) <= 2 and abs(ring[:, 1].max() - 455) <= 2
    again = dm.clip_marks(DAY, 'c20260917_101030', shop)
    assert list((shop / 'data/day_movie' / DAY / 'outlines').glob('c20260917_101030_*.json.gz'))   # kept on disk
    assert json.loads(gzip.decompress(again)) == body


def test_a_bridge_between_two_pieces_of_a_mask_is_cut_away_and_the_pieces_stay():
    """Two blocks of one mask (head and body split by a stand), joined the way Ultralytics
    joins pieces and then thinned by taking every other point: the bridge became a spike."""
    head = [[1000, 200], [1100, 200], [1100, 300], [1000, 300]]
    body = [[980, 400], [1120, 400], [1120, 900], [980, 900]]
    # out along x=1051, back along x=1049: a zero-width bridge thinned apart by about 2 px
    joined = [head[0], head[1], [1100, 300], [1051, 300], [1051, 400], body[1], body[2], body[3], body[0],
              [1049, 400], [1049, 300], head[3]]
    rings = dm.clean_outline(np.array(joined, np.float32), unit=1280)
    assert len(rings) == 2
    tops = sorted(r[:, 1].min() for r in rings)
    assert abs(tops[0] - 100) <= 2 and abs(tops[1] - 200) <= 2     # the head and the body, at half size
    between = [r for r in rings if r[:, 1].min() < 160 and r[:, 1].max() > 190]
    assert not between                                              # nothing left spanning the gap


def test_the_machine_slows_down_where_somebody_vanished_mid_room_not_at_a_seam(shop):
    moments = dm.doubts(DAY, shop)
    last = 30 + float(np.arange(100, 160, 0.12)[-1])
    assert any(abs(m - last) < 0.02 for m in moments)                         # B stopped mid-room
    assert not any(abs(m - (30 + 599.88)) < 1 for m in moments)             # A at the seam: carried on


def _rev(root):
    return dm.store(DAY, root)['revision']


def test_split_join_false_staff_missed_and_undo(shop):
    crowd = dm.people(DAY, shop)
    a = crowd['parts']['c20260917_100030:0@']['person']
    did = dm.change(DAY, {'action': 'split', 'revision': 0, 'part': 'c20260917_101030:0@', 'at': 650.0, 'rate': 1}, shop)
    assert did['part'] == 'c20260917_101030:0@650.00'
    parts = dm.people(DAY, shop)['parts']
    assert parts['c20260917_101030:0@']['person'] == a
    assert parts['c20260917_101030:0@650.00']['person'] not in (a, None)
    # the owner says the second half is B after all
    b_part = 'c20260917_100030:2@'
    dm.change(DAY, {'action': 'same', 'revision': _rev(shop), 'part': 'c20260917_101030:0@650.00', 'target': b_part}, shop)
    parts = dm.people(DAY, shop)['parts']
    assert parts['c20260917_101030:0@650.00']['person'] == parts[b_part]['person']
    with pytest.raises(ValueError, match='уже один'):
        dm.change(DAY, {'action': 'same', 'revision': _rev(shop), 'part': b_part, 'target': 'c20260917_101030:0@650.00'}, shop)
    dm.change(DAY, {'action': 'false', 'revision': _rev(shop), 'part': 'c20260917_100030:1@'}, shop)
    parts = dm.people(DAY, shop)['parts']
    assert parts['c20260917_100030:1@']['false'] and parts['c20260917_100030:1@']['person'] is None
    staff = dm.change(DAY, {'action': 'staff', 'revision': _rev(shop), 'part': 'c20260917_100030:0@'}, shop)
    assert staff['now'] and dm.people(DAY, shop)['persons'][a]['staff']
    unstaff = dm.change(DAY, {'action': 'staff', 'revision': _rev(shop), 'part': 'c20260917_101030:0@'}, shop)
    assert not unstaff['now'] and not dm.people(DAY, shop)['persons'][a]['staff']   # the same person, any part
    mark = dm.change(DAY, {'action': 'missed', 'revision': _rev(shop), 'cam': 'cam1', 'at': 700, 'x': 300, 'y': 200}, shop)
    again = dm.change(DAY, {'action': 'missed', 'revision': _rev(shop), 'cam': 'cam1', 'at': 700.5, 'x': 305, 'y': 210}, shop)
    assert mark['now'] and not again['now'] and dm.store(DAY, shop)['missed'] == []
    dm.change(DAY, {'action': 'undo', 'revision': _rev(shop)}, shop)
    assert len(dm.store(DAY, shop)['missed']) == 1
    with pytest.raises(ValueError, match='Отменять нечего'):
        dm.change(DAY, {'action': 'undo', 'revision': _rev(shop)}, shop)
    with pytest.raises(ValueError, match='другой вкладке'):
        dm.change(DAY, {'action': 'false', 'revision': 0, 'part': b_part}, shop)


def test_s_at_the_start_of_a_track_gives_the_whole_track_away(shop):
    did = dm.change(DAY, {'action': 'split', 'revision': 0, 'part': 'c20260917_101030:0@', 'at': 630.5, 'rate': 1}, shop)
    assert did['part'] == 'c20260917_101030:0@'
    crowd = dm.people(DAY, shop)
    assert crowd['parts']['c20260917_101030:0@']['person'] != crowd['parts']['c20260917_100030:0@']['person']


def test_at_speed_the_cut_goes_where_the_look_changed_not_where_the_key_was_pressed(tmp_path, monkeypatch):
    monkeypatch.setattr(dp, '_raw_segments', lambda cam, day: [])
    dp._cache.clear(); dm._cache.clear()
    root = tmp_path / 'p'
    _segments(root)
    atomic_json(root / 'data/cam_sync.json', {DAY: {'cam1_to_cam2_s': 0.0}})
    moments = list(np.arange(100, 130, 0.12))
    switch = moments.index(next(t for t in moments if t >= 118.0))
    _clip(root, 'c20260917_100030', T0 + timedelta(seconds=30), [('cam1', moments, MID)], groups=[[0]],
          looks={0: lambda n: 0 if n < switch else 7})
    # pressed at 8x a good second and a half later than the switch (film time = 30 + clip time)
    did = dm.change(DAY, {'action': 'split', 'revision': 0, 'part': 'c20260917_100030:0@', 'at': 30 + 121.0, 'rate': 8}, root)
    cut = float(did['part'].split('@')[1])
    assert abs(cut - (30 + 118.0)) < 0.2, cut


def test_edits_on_a_track_the_night_recomputed_stop_counting(shop):
    dm.change(DAY, {'action': 'false', 'revision': 0, 'part': 'c20260917_100030:2@'}, shop)
    assert dm.people(DAY, shop)['parts']['c20260917_100030:2@']['false']
    pieces = read_json(shop / 'data/raw_clips/c20260917_100030/pieces_yolo26x-seg.json')
    pieces[2]['dets'] = pieces[2]['dets'][:-5]
    atomic_json(shop / 'data/raw_clips/c20260917_100030/pieces_yolo26x-seg.json', pieces)
    crowd = dm.people(DAY, shop)
    assert crowd['stale'] == ['c20260917_100030:2'] and not crowd['parts']['c20260917_100030:2@']['false']


def test_watched_is_kept_with_its_speed_and_nonsense_is_dropped(shop):
    out = dm.add_watched(DAY, [[10, 20, 8], [19.9, 30, 2], [50, 40, 8], [0, 1000, 8], 'x'], shop)
    assert out['spans'] == [[10.0, 30.0]] and out['seconds'] == 20.0 and out['fastest'] == 8
    assert len(read_json(dm._watched_path(DAY, shop))['spans']) == 2


def test_routes(shop):
    from flask import Flask
    app = Flask(__name__, template_folder=str(Path(dm.__file__).parent / 'templates'))
    dm.register(app, lambda: shop)
    client = app.test_client()
    view = client.get('/api/movie/%s' % DAY).get_json()
    assert view['videos']['cam1']['ready'] is False and view['duration'] == 2700
    assert len(view['persons']) == 2 and view['revision'] == 0
    assert client.get('/api/movie/%s/clip/c20260917_101030' % DAY).headers['Content-Encoding'] == 'gzip'
    changed = client.post('/api/movie/%s' % DAY, json={'action': 'false', 'revision': 0, 'part': 'c20260917_100030:2@'})
    assert changed.status_code == 200 and changed.get_json()['did']['now'] is True
    stale = client.post('/api/movie/%s' % DAY, json={'action': 'false', 'revision': 0, 'part': 'c20260917_100030:2@'})
    assert stale.status_code == 409
    assert client.get('/movie-file/%s/cam1.mp4' % DAY).status_code == 404
    assert client.get('/api/movie/%s/clip/c_nowhere' % DAY).status_code == 404
    assert client.get('/api/movie/abc').status_code == 400
    assert client.get('/movie').status_code == 200
    assert client.post('/api/movie/%s/watched' % DAY, json={'spans': [[1, 2, 4]]}).get_json()['seconds'] == 1.0


def test_numbers_do_not_move_when_the_owner_edits(shop):
    before = {p['n'] for p in dm.people(DAY, shop)['persons'].values() if 'n' in p}
    crowd = dm.people(DAY, shop)
    a = crowd['parts']['c20260917_100030:0@']['person']
    assert before == {1, 2} and crowd['persons'][a]['n'] == 2
    dm.change(DAY, {'action': 'false', 'revision': 0, 'part': 'c20260917_100030:2@'}, shop)    # #1 is not a person
    crowd = dm.people(DAY, shop)
    assert crowd['persons'][a]['n'] == 2                   # A is still #2, not renumbered to #1
    dm.change(DAY, {'action': 'split', 'revision': 1, 'part': 'c20260917_101030:0@', 'at': 650.0, 'rate': 1}, shop)
    crowd = dm.people(DAY, shop)
    new = crowd['parts']['c20260917_101030:0@650.00']['person']
    assert crowd['persons'][a]['n'] == 2 and crowd['persons'][new]['n'] == 3
    assert crowd['persons'][new]['color'] != crowd['persons'][a]['color']


def test_a_track_handed_to_somebody_else_is_found_by_its_look_and_slowed_down_at(tmp_path, monkeypatch):
    monkeypatch.setattr(dp, '_raw_segments', lambda cam, day: [])
    dp._cache.clear(); dm._cache.clear()
    root = tmp_path / 'p'
    _segments(root)
    atomic_json(root / 'data/cam_sync.json', {DAY: {'cam1_to_cam2_s': 0.0}})
    moments = list(np.arange(100, 130, 0.12))
    switch = moments.index(next(t for t in moments if t >= 118.0))
    _clip(root, 'c20260917_100030', T0 + timedelta(seconds=30), [('cam1', moments, MID), ('cam1', list(np.arange(0, 30, 0.12)), OTHER)],
          groups=[[0], [1]], looks={0: lambda n: 0 if n < switch else 7})
    found = dm.switches(DAY, root, log=lambda s: None)
    strong = [f for f in found if f[2] >= dm.SWITCH]
    assert len(strong) == 1 and strong[0][1] == 'c20260917_100030:0' and abs(strong[0][0] - 148.0) < 0.2
    assert any(abs(m - 148.0) < 0.2 for m in dm.doubts(DAY, root))
    dm.change(DAY, {'action': 'split', 'revision': 0, 'part': 'c20260917_100030:0@', 'at': 148.0, 'rate': 1}, root)
    assert not any(abs(m - 148.0) < 1 for m in dm.doubts(DAY, root))      # the owner cut it: no longer a doubt


def test_passers_by_are_outlined_but_numbered_only_when_the_owner_says_they_came_in(shop):
    _clip(shop, 'c20260917_102030', T0 + timedelta(seconds=1230), [('cam1', list(np.arange(10, 15, 0.12)), OTHER)])
    dm._cache.clear()
    crowd = dm.people(DAY, shop)
    passer = crowd['parts']['c20260917_102030:0@']['person']
    assert crowd['persons'][passer]['kind'] == 'passer' and crowd['persons'][passer]['n'] is None
    rev = dm.store(DAY, shop)['revision']
    dm.change(DAY, {'action': 'kind', 'revision': rev, 'part': 'c20260917_102030:0@', 'kind': 'customer'}, shop)
    crowd = dm.people(DAY, shop)
    assert crowd['persons'][passer]['kind'] == 'customer' and crowd['persons'][passer]['n'] == 3
    dm.change(DAY, {'action': 'kind', 'revision': rev + 1, 'part': 'c20260917_102030:0@', 'kind': 'staff'}, shop)
    assert dm.people(DAY, shop)['persons'][passer]['kind'] == 'staff'
    assert list(dm.store(DAY, shop)['kinds'].values()) == ['staff']              # one word per person, the latest
    with pytest.raises(ValueError, match='Неизвестно'):
        dm.change(DAY, {'action': 'kind', 'revision': rev + 2, 'part': 'c20260917_102030:0@', 'kind': 'boss'}, shop)
    view = dm.overview(DAY, shop)
    assert view['persons'][passer]['kind'] == 'staff' and view['activity'][-1][1] >= 1230 + 14


def test_a_bad_outline_is_marked_and_taken_back(shop):
    on = dm.change(DAY, {'action': 'badmask', 'revision': 0, 'part': 'c20260917_100030:0@', 'at': 540.0}, shop)
    assert on['now'] and dm.store(DAY, shop)['badmask'][0]['cam'] == 'cam1'
    off = dm.change(DAY, {'action': 'badmask', 'revision': 1, 'part': 'c20260917_100030:0@', 'at': 540.8}, shop)
    assert not off['now'] and dm.store(DAY, shop)['badmask'] == []


def test_a_store_written_by_the_first_release_still_reads(shop):
    atomic_json(dm._store_path(DAY, shop), {'revision': 3, 'undo_revision': 2, 'cuts': {}, 'same': [], 'detached': [],
                                            'false': [], 'staff': ['c20260917_100030:0@'], 'missed': [], 'prints': {},
                                            'next_mark': 1})
    crowd = dm.people(DAY, shop)
    a = crowd['parts']['c20260917_100030:0@']['person']
    assert crowd['persons'][a]['kind'] == 'staff' and dm.store(DAY, shop)['badmask'] == []


def test_the_half_cut_off_keeps_its_own_number_when_the_first_half_is_merged_away(shop):
    dm.change(DAY, {'action': 'split', 'revision': 0, 'part': 'c20260917_101030:0@', 'at': 650.0, 'rate': 1}, shop)
    crowd = dm.people(DAY, shop)
    b_half = crowd['parts']['c20260917_101030:0@650.00']['person']
    assert crowd['persons'][b_half]['n'] == 3
    # A (#2) turns out to be B (#1): merged away, number 2 is free now
    dm.change(DAY, {'action': 'same', 'revision': 1, 'part': 'c20260917_100030:0@', 'target': 'c20260917_100030:2@'}, shop)
    crowd = dm.people(DAY, shop)
    assert crowd['persons'][crowd['parts']['c20260917_101030:0@650.00']['person']]['n'] == 3


@pytest.mark.skipif(not FFMPEG, reason='needs ffmpeg')
def test_the_frame_the_detector_really_started_on_is_replayed_and_used(tmp_path, monkeypatch):
    """OpenCV seeks CAP_PROP_POS_FRAMES by timestamp: in a segment that lost 2 s at frame 500,
    asking for frame 600 does not give frame 600. The clip's detections must be timed from
    the frame it did give, or every outline of the clip is off by the drop."""
    import cv2
    root = tmp_path / 'p'
    raw = tmp_path / 'raw' / '100000_0000.mp4'
    raw.parent.mkdir(parents=True)
    plain = tmp_path / 'plain.mp4'
    pipe = subprocess.Popen([FFMPEG, '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-s', '640x360',
                             '-r', '25', '-i', '-', '-c:v', 'libx264', '-bf', '0', '-g', '50', '-crf', '12',
                             '-pix_fmt', 'yuv420p', str(plain)], stdin=subprocess.PIPE)
    for k in range(1200):
        pipe.stdin.write(_pattern(k + 1).tobytes())
    pipe.stdin.close(); assert pipe.wait() == 0
    subprocess.run([FFMPEG, '-y', '-loglevel', 'error', '-i', str(plain), '-vf', "setpts='if(gte(N,500),PTS+2/TB,PTS)'",
                    '-fps_mode', 'passthrough', '-c:v', 'libx264', '-bf', '0', '-g', '50', '-crf', '12',
                    '-pix_fmt', 'yuv420p', str(raw)], check=True)
    for cam in ('cam1', 'cam2'):
        folder = root / 'data/day_proxy' / DAY / cam
        folder.mkdir(parents=True)
        (folder / '100000_0000.mp4').write_bytes(b'copy')
        atomic_json(folder / '100000_0000.json', {'day': DAY, 'cam': cam, 'raw': '100000_0000.mp4', 'start': T0.isoformat(),
                                                  'frames': 1200, 'breaks': [[0, 0.0], [500, 22.0]]})
    monkeypatch.setattr(dp, '_raw_segments', lambda cam, day: [(str(raw), T0)])
    dp._cache.clear()
    segs = dp.segments(DAY, root)
    clip = root / 'data/raw_clips/c1'
    clip.mkdir(parents=True)
    landed = dp.start_frames(clip, {'start': (T0 + timedelta(seconds=24)).isoformat()}, segs, DAY)
    nominal = 600
    assert landed['cam1'] is not None and landed['cam1'] != nominal
    # the frame OpenCV hands over is the frame the replay names
    capture = cv2.VideoCapture(str(raw)); capture.set(cv2.CAP_PROP_POS_FRAMES, nominal)
    ok, image = capture.read(); capture.release()
    assert _read_pattern(image) - 1 == landed['cam1']
    assert read_json(clip / 'start_frames.json')['cam1']['nominal'] == nominal       # kept beside the clip
    # and the clock of the clip's frames starts there, not at the nominal index
    at = dp.walls(segs['cam1'], T0.timestamp() + 24, [0, 10], landed['cam1']) - T0.timestamp()
    times = segs['cam1'][0].times()
    assert list(np.round(at, 2)) == [round(times[landed['cam1']], 2), round(times[landed['cam1'] + 10], 2)]


def test_a_new_camera_offset_keeps_the_film_clock_and_asks_for_camera_two_again(shop):
    start, end = dm.clock(DAY, shop)
    for cam, shift in (('cam1', 0.0), ('cam2', 0.0)):
        mp4, side = dm.film(DAY, cam, shop)
        mp4.parent.mkdir(parents=True, exist_ok=True)
        mp4.write_bytes(b'film')
        atomic_json(side, {'day': DAY, 'cam': cam, 'start': start, 'end': end, 'shift': shift, 'holes': [],
                           'version': dm.VERSION})
    assert dm.overview(DAY, shop)['videos']['cam2']['ready']
    atomic_json(dm.folder(DAY, shop) / 'offset.json', {'cam1_to_cam2_s': -0.8})
    dp._cache.clear(); dm._cache.clear()
    assert dm.clock(DAY, shop) == (start, end)
    view = dm.overview(DAY, shop)
    assert view['videos']['cam1']['ready'] and view['videos']['cam2']['stale'] and not view['videos']['cam2']['ready']
    assert view['offset'] == -0.8


def test_detections_the_tracker_left_out_are_outlined_too(shop):
    """The film drew pieces only, so a person the detector found but the tracker dropped had
    no outline. Left-out boxes are chained by overlap; a box lying on a kept one is its double."""
    folder = shop / 'data/raw_clips/c20260917_101030'
    with np.load(folder / 'dets_yolo26x-seg.npz') as archive:
        d = {c: archive[c] for c in ('cam1', 'cam2')}
    extra = []
    for t in list(np.arange(20, 22, 0.12)) + list(np.arange(30, 30.5, 0.12)):   # a walker, then after a gap
        row = np.zeros(10, np.float32); row[0] = t; row[1:5] = [2000, 300, 2150, 800]; extra.append(row)
    double = np.zeros(10, np.float32); double[0] = d['cam1'][5, 0]; double[1:5] = d['cam1'][5, 1:5] + [4, 4, 4, 4]
    extra.append(double)
    d['cam1'] = np.vstack([d['cam1'], np.array(extra, np.float32)])
    np.savez(folder / 'dets_yolo26x-seg.npz', **d)
    dp._cache.clear(); dm._cache.clear()
    tracks = dm.index(DAY, shop)['tracks']
    extra_keys = sorted(k for k in tracks if ':x' in k)
    assert extra_keys == ['c20260917_101030:x1.0', 'c20260917_101030:x1.1']        # the gap splits, the double is gone
    first = tracks['c20260917_101030:x1.0']
    assert len(first['rows']) == len(np.arange(20, 22, 0.12)) and first['first'] == pytest.approx(630 + 20, abs=0.01)
    crowd = dm.people(DAY, shop)
    walker = crowd['parts']['c20260917_101030:x1.0@']['person']
    assert crowd['persons'][walker]['kind'] == 'passer'                               # short, never on the floor
    body = json.loads(gzip.decompress(dm.clip_marks(DAY, 'c20260917_101030', shop)))
    assert {'c20260917_101030:x1.0', 'c20260917_101030:x1.1'} <= {t['k'] for t in body['tracks']}
