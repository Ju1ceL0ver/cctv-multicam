"""The whole-day labelling tool: true frame times, tracks on the shop clock, the owner's store."""
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'multicam'))
import day_player as dp                              # noqa: E402
from storage import atomic_json, read_json           # noqa: E402

DAY = '20260917'
T0 = datetime(2026, 9, 17, 10, 0, 0)


def _clip(root, name, start, people):
    """people: [(cam, [seconds...])] or [(cam, [seconds...], who)] -- one machine track each.
    `who` is a small integer: tracks sharing it get the same appearance descriptor."""
    d = root / 'data/raw_clips' / name
    d.mkdir(parents=True)
    rows = {'cam1': [], 'cam2': []}
    looks = {'cam1': [], 'cam2': []}
    pieces = []
    for index, person in enumerate(people):
        cam, moments = person[0], person[1]
        who = person[2] if len(person) > 2 else index
        face = np.zeros(512, np.float32)
        face[who % 512] = 1.0
        first = len(rows[cam])
        for t in moments:
            row = np.zeros(10, np.float32)
            row[0] = t
            row[1:5] = [100 + index * 300, 200, 250 + index * 300, 600]
            rows[cam].append(row)
            looks[cam].append(face)
        pieces.append({'piece': index, 'cam': cam, 't0': float(moments[0]), 't1': float(moments[-1]),
                       'dets': list(range(first, len(rows[cam])))})
    atomic_json(d / 'meta_yolo26x-seg.json', {'day': DAY, 'start': start.isoformat(), 'seconds': 600})
    atomic_json(d / 'pieces_yolo26x-seg.json', pieces)
    np.savez(d / 'dets_yolo26x-seg.npz',
             **{c: (np.array(v, np.float32) if v else np.zeros((0, 10), np.float32)) for c, v in rows.items()})
    np.savez(d / 'emb_osnet_ain_x1_0_msmt17.npz',
             **{c: (np.array(v, np.float32) if v else np.zeros((0, 512), np.float32)) for c, v in looks.items()})
    return d


@pytest.fixture
def day(tmp_path, monkeypatch):
    """Camera 1 lost 50 frames in one burst at 10 s (the file jumps from 9.96 s to 12.00 s),
    so its 15-minute segment holds 22450 frames. Camera 2 has no light copy yet."""
    root = tmp_path / 'project'
    side = root / 'data/day_proxy' / DAY / 'cam1'
    side.mkdir(parents=True)
    atomic_json(side / '100000_0000.json', {'day': DAY, 'cam': 'cam1', 'raw': '100000_0000.mp4',
                                             'start': T0.isoformat(), 'frames': 22450,
                                             'breaks': [[0, 0.0], [250, 12.0]]})
    (side / '100000_0000.mp4').write_bytes(b'not a real video')
    atomic_json(root / 'data/cam_sync.json', {DAY: {'cam1_to_cam2_s': 4.0}})
    _clip(root, 'c_a', T0, [('cam1', [0, 9.96, 10.0, 20]),         # frame 250 is at 12.0 s, not 10.0
                            ('cam2', [6, 24]),
                            ('cam1', [30, 33])])
    monkeypatch.setattr(dp, '_raw_segments', lambda cam, day: [('C:/raw/%s/100000_0000.mp4' % cam, T0)])
    dp._cache.clear()
    return root


def test_a_frame_after_a_drop_gets_its_true_time_not_the_counted_one(day):
    segs = dp.segments(DAY, day)['cam1']
    at = dp.walls(segs, T0.timestamp(), [249, 250, 300])
    assert list(np.round(at - T0.timestamp(), 2)) == [9.96, 12.0, 14.0]


def test_a_clip_that_runs_past_a_short_segment_continues_in_the_next_one(day):
    segs = [dp.Segment('cam1', 'a', 0.0, frames=100), dp.Segment('cam1', 'b', 900.0)]
    at = dp.walls(segs, 2.0, [0, 49, 50, 60])       # starts at index 50 of a 100-frame segment
    assert list(np.round(at, 2)) == [2.0, 3.96, 900.0, 900.4]


def test_tracks_are_on_camera_one_clock_and_camera_two_is_shifted_back(day):
    rows = {r['key']: r for r in dp.tracks(DAY, day)['tracks']}
    base = T0.timestamp()
    assert rows['c_a:0']['first'] - base == 0.0 and round(rows['c_a:0']['last'] - base, 2) == 22.0
    # camera 2 sees at 10:00:06 what camera 1 saw at 10:00:02
    assert round(rows['c_a:1']['first'] - base, 2) == 2.0 and rows['c_a:1']['cam'] == 'cam2'
    boxes = dp.clip_boxes(DAY, 'c_a', day)['c_a:0']
    assert [round(b[0] - base, 2) for b in boxes] == [0.0, 9.96, 12.0, 22.0]
    assert boxes[0][1:] == [100, 200, 250, 600]


def test_the_owner_names_a_person_and_the_whole_track_is_theirs(day):
    made = dp.change(DAY, {'action': 'person', 'revision': 0, 'name': 'синяя олимпийка', 'kind': 'customer'}, day)
    assert 'P1' in made['people']
    done = dp.change(DAY, {'action': 'assign', 'revision': made['revision'], 'track': 'c_a:0', 'who': 'P1'}, day)
    assert done['changed']['c_a:0']['who'] == ['P1']
    bulk = dp.change(DAY, {'action': 'assign_many', 'revision': done['revision'], 'who': '_passer',
                           'parts': [['c_a:2', 0]], 'rule': 'short_sweep'}, day)
    stored = dp.store(DAY, day)
    assert stored['tracks']['c_a:2']['who'] == ['_passer'] and stored['tracks']['c_a:2']['rules'] == {'0': 'short_sweep'}
    assert bulk['revision'] == 3


def test_a_track_holding_two_people_is_cut_and_each_half_named(day):
    dp.change(DAY, {'action': 'person', 'revision': 0, 'name': 'A', 'kind': 'customer'}, day)
    dp.change(DAY, {'action': 'person', 'revision': 1, 'name': 'B', 'kind': 'staff'}, day)
    dp.change(DAY, {'action': 'assign', 'revision': 2, 'track': 'c_a:0', 'who': 'P1'}, day)
    cut = dp.change(DAY, {'action': 'cut', 'revision': 3, 'track': 'c_a:0', 'at': T0.timestamp() + 11}, day)
    assert cut['changed']['c_a:0']['who'] == ['P1', 'P1']            # both halves start as they were
    dp.change(DAY, {'action': 'assign', 'revision': 4, 'track': 'c_a:0', 'part': 1, 'who': 'P2'}, day)
    record = dp.store(DAY, day)['tracks']['c_a:0']
    assert record['who'] == ['P1', 'P2'] and len(record['cuts']) == 1
    with pytest.raises(ValueError, match='только внутри'):
        dp.change(DAY, {'action': 'cut', 'revision': 5, 'track': 'c_a:0', 'at': T0.timestamp() + 60}, day)


def test_undo_goes_back_one_edit_and_not_further(day):
    dp.change(DAY, {'action': 'person', 'revision': 0, 'name': 'A', 'kind': 'customer'}, day)
    dp.change(DAY, {'action': 'assign', 'revision': 1, 'track': 'c_a:0', 'who': 'P1'}, day)
    back = dp.change(DAY, {'action': 'undo', 'revision': 2}, day)
    assert 'c_a:0' not in back['store']['tracks'] and back['revision'] == 3
    with pytest.raises(ValueError, match='нечего'):
        dp.change(DAY, {'action': 'undo', 'revision': 3}, day)


@pytest.mark.parametrize('body,message', [
    ({'action': 'assign', 'revision': 7, 'track': 'c_a:0', 'who': '_passer'}, 'другой вкладке'),
    ({'action': 'assign', 'revision': 0, 'track': 'c_zz:9', 'who': '_passer'}, 'не существует'),
    ({'action': 'assign', 'revision': 0, 'track': 'c_a:0', 'who': 'P77'}, 'Нет такого'),
    ({'action': 'assign', 'revision': 0, 'track': 'c_a:0', 'part': 3, 'who': '_passer'}, 'части'),
    ({'action': 'person', 'revision': 0, 'name': '', 'kind': 'customer'}, 'имя'),
    ({'action': 'person', 'revision': 0, 'name': 'X', 'kind': 'boss'}, 'имя'),
    ({'action': 'dance', 'revision': 0}, 'Неизвестное'),
])
def test_a_wrong_edit_is_refused_and_changes_nothing(day, body, message):
    with pytest.raises(ValueError, match=message):
        dp.change(DAY, body, day)
    assert dp.store(DAY, day)['revision'] == 0


def test_a_track_recomputed_at_night_is_flagged_not_silently_relabelled(day):
    dp.change(DAY, {'action': 'assign', 'revision': 0, 'track': 'c_a:0', 'who': '_passer'}, day)
    atomic_json(day / 'data/raw_clips/c_a/pieces_yolo26x-seg.json',
                [{'piece': 0, 'cam': 'cam1', 't0': 0.0, 't1': 9.96, 'dets': [0, 1]},
                 {'piece': 1, 'cam': 'cam2', 't0': 6.0, 't1': 24.0, 'dets': [0, 1]},
                 {'piece': 2, 'cam': 'cam1', 't0': 30.0, 't1': 33.0, 'dets': [4, 5]}])
    assert dp.overview(DAY, day)['stale'] == ['c_a:0']
    with pytest.raises(ValueError, match='пересчитан'):
        dp.change(DAY, {'action': 'assign', 'revision': 1, 'track': 'c_a:0', 'who': '_false'}, day)


def test_the_page_its_data_and_the_video_copies_are_served(day, monkeypatch):
    import label_pieces as web
    monkeypatch.setattr(web, 'ROOT', str(day))
    monkeypatch.setattr(web, 'CLIPS', str(day / 'data/raw_clips'))
    web.app.config['TESTING'] = True
    client = web.app.test_client()
    client.set_cookie('labeler_key', web.KEY)
    assert client.get('/day').status_code == 200
    assert client.get('/static/day.js').status_code == 200
    data = client.get('/api/dayplayer/%s' % DAY).json
    assert len(data['tracks']) == 3 and data['offset'] == 4.0
    ready = {s['name']: s['ready'] for s in data['segments']['cam1']}
    assert ready == {'100000_0000.mp4': True}
    assert data['segments']['cam2'][0]['ready'] is False
    assert set(client.get('/api/dayplayer/%s/boxes?clip=c_a' % DAY).json) == {'c_a:0', 'c_a:1', 'c_a:2'}
    assert client.get('/day-proxy/%s/cam1/100000_0000.mp4' % DAY).status_code == 200
    assert client.get('/day-proxy/%s/cam2/100000_0000.mp4' % DAY).status_code == 404
    assert client.get('/day-proxy/%s/cam1/../../x.mp4' % DAY).status_code == 404
    saved = client.post('/api/dayplayer/%s' % DAY, json={'action': 'assign', 'revision': 0,
                                                          'track': 'c_a:1', 'who': '_false'})
    assert saved.status_code == 200 and saved.json['revision'] == 1
    assert client.post('/api/dayplayer/%s' % DAY, json={'action': 'assign', 'revision': 0,
                                                        'track': 'c_a:1', 'who': '_false'}).status_code == 409


def test_a_recording_the_machine_never_finished_is_called_broken_not_pending(day):
    """17.09 ends with a segment the reboot cut off: ffmpeg finds no moov atom. The owner
    must be told it is broken, not that a copy is still being made."""
    side = day / 'data/day_proxy' / DAY / 'cam1' / '100000_0001.json'
    atomic_json(side, {'day': DAY, 'cam': 'cam1', 'raw': '100000_0001.mp4',
                       'start': datetime(2026, 9, 17, 10, 15, 0).isoformat(),
                       'frames': 0, 'error': 'ffmpeg: moov atom not found'})
    dp._cache.clear()
    found = {s.name: s for s in dp.segments(DAY, day)['cam1']}
    assert found['100000_0001.mp4'].broken and not found['100000_0001.mp4'].ready
    shown = {s['name']: s for s in dp.overview(DAY, day)['segments']['cam1']}
    assert shown['100000_0001.mp4']['broken'] is True


@pytest.fixture
def measured(day, monkeypatch):
    """The shop floor as a simple band, so the test is about the code and not about where
    somebody traced the polygon: only feet between x=300 and x=600 are inside the shop.
    That leaves the first track (feet at 175) and the short third one (at 775) outside,
    exactly as the gallery behind the glass is."""
    import day_features
    import rooms
    def mask(cam, shape=(1440, 2560)):
        m = np.zeros(shape, np.uint8)
        m[:, 300:600] = 255
        return m
    monkeypatch.setattr(rooms, 'mask', mask)
    day_features.build(DAY, day, log=lambda *a: None)
    dp._cache.clear()
    return day


def test_a_track_that_never_stood_in_the_shop_is_told_apart(measured):
    rows = {r['key']: r for r in dp.tracks(DAY, measured)['tracks']}
    assert rows['c_a:0']['floor'] == 0.0          # feet at 175: behind the glass
    assert rows['c_a:1']['floor'] == 1.0          # feet at 475: inside the shop
    assert rows['c_a:2']['floor'] == 0.0          # feet at 775: behind the glass again
    assert dp.overview(DAY, measured)['measured'] is True


def test_the_gallery_is_swept_in_one_go_and_says_by_which_rule(measured):
    done = dp.change(DAY, {'action': 'sweep_gallery', 'revision': 0, 'seconds': 10}, measured)
    swept = dp.store(DAY, measured)['tracks']
    assert list(swept) == ['c_a:2']               # short and never on the floor
    assert swept['c_a:2']['who'] == ['_passer'] and swept['c_a:2']['rules'] == {'0': 'gallery_not_on_floor'}
    assert 'c_a:2' in done['changed']
    with pytest.raises(ValueError, match='не осталось'):
        dp.change(DAY, {'action': 'sweep_gallery', 'revision': 1}, measured)


def test_naming_one_person_offers_every_other_track_that_looks_like_them(tmp_path, monkeypatch):
    root = tmp_path / 'project'
    (root / 'data/day_proxy' / DAY / 'cam1').mkdir(parents=True)
    atomic_json(root / 'data/cam_sync.json', {DAY: {'cam1_to_cam2_s': 0.0}})
    #            the same person twice on camera 1, once on camera 2 at the same moment, and a stranger
    _clip(root, 'c_a', T0, [('cam1', [0, 20], 7), ('cam1', [40, 60], 7),
                            ('cam2', [0, 20], 7), ('cam1', [80, 100], 3)])
    monkeypatch.setattr(dp, '_raw_segments', lambda cam, day: [('C:/raw/%s/100000_0000.mp4' % cam, T0)])
    import day_features, rooms
    monkeypatch.setattr(rooms, 'mask', lambda cam, shape=(1440, 2560): np.full(shape, 255, np.uint8))
    day_features.build(DAY, root, log=lambda *a: None)
    dp._cache.clear()
    found = dp.similar(DAY, 'c_a:0', root)
    assert found['ready'] and found['accept'] == 0.228
    by = {c['key']: c for c in found['candidates']}
    assert by['c_a:1']['suggested'] and by['c_a:1']['distance'] == 0.0
    # same camera, same moment: one person cannot be in two places at once
    assert by['c_a:2']['distance'] == 0.0 and by['c_a:2']['cam'] == 'cam2' and by['c_a:2']['suggested']
    assert by['c_a:3']['distance'] == 1.0 and not by['c_a:3']['suggested']
    dp.change(DAY, {'action': 'person', 'revision': 0, 'name': 'X', 'kind': 'customer'}, root)
    dp.change(DAY, {'action': 'assign', 'revision': 1, 'track': 'c_a:1', 'who': 'P1'}, root)
    again = dp.similar(DAY, 'P1', root)            # by person now, and the taken track is gone
    assert 'c_a:1' not in {c['key'] for c in again['candidates']}
    assert {c['key'] for c in again['candidates']} == {'c_a:0', 'c_a:2', 'c_a:3'}


def test_a_candidate_seen_at_the_same_time_on_the_same_camera_is_impossible(tmp_path, monkeypatch):
    root = tmp_path / 'project'
    atomic_json(root / 'data/cam_sync.json', {DAY: {'cam1_to_cam2_s': 0.0}})
    _clip(root, 'c_a', T0, [('cam1', [0, 20], 7), ('cam1', [5, 25], 7)])
    monkeypatch.setattr(dp, '_raw_segments', lambda cam, day: [('C:/raw/cam1/100000_0000.mp4', T0)])
    import day_features, rooms
    monkeypatch.setattr(rooms, 'mask', lambda cam, shape=(1440, 2560): np.full(shape, 255, np.uint8))
    day_features.build(DAY, root, log=lambda *a: None)
    dp._cache.clear()
    twin = dp.similar(DAY, 'c_a:0', root)['candidates'][0]
    assert twin['key'] == 'c_a:1' and twin['distance'] == 0.0
    assert twin['impossible'] and not twin['suggested']
