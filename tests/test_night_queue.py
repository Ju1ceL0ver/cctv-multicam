"""The night queue must not be held up by one window that cannot be labelled.

On 20.09.2026 the last window of 17.09 (19:00, cut short by the reboot) crashed the
synchronisation step every night. The keeper always went back to the oldest day that was not
fully covered, so 18, 19 and 20.09 were never labelled and the card idled all night."""
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'multicam'))
_cwd = os.getcwd()
import clip_registry as registry     # noqa: E402
import keeper                        # noqa: E402  (its import changes the working directory)
import auto_label                    # noqa: E402
import person3d                      # noqa: E402
os.chdir(_cwd)

DAY_A, DAY_B = '20260917', '20260918'


def hm(day, hh, mm):
    return datetime.strptime('%s%02d%02d54' % (day, hh, mm), '%Y%m%d%H%M%S')


@pytest.fixture
def box(tmp_path, monkeypatch):
    """A keeper root with two recorded days: A has three windows (two ready), B has two."""
    windows = {DAY_A: [hm(DAY_A, 10, 0), hm(DAY_A, 16, 0), hm(DAY_A, 19, 0)],
               DAY_B: [hm(DAY_B, 10, 0), hm(DAY_B, 16, 0)]}
    monkeypatch.setattr(keeper, 'ROOT', str(tmp_path))
    monkeypatch.setattr(keeper, 'windows', lambda day, minutes: windows[day])
    monkeypatch.setattr(keeper, 'recorded_days', lambda: [DAY_A, DAY_B])
    monkeypatch.setattr(keeper, 'still_recording', lambda day: False)
    for t in windows[DAY_A][:2]:
        folder = tmp_path / 'data' / 'raw_clips' / registry.clip_key(DAY_A, t, tmp_path / 'data' / 'raw_clips')
        for name in ('people', 'pieces', 'groups'):
            (folder).mkdir(parents=True, exist_ok=True)
            (folder / (name + '_yolo26x-seg.json')).write_text('[]')
    return tmp_path, windows


def test_a_window_steps_aside_after_repeated_failures_and_comes_back_next_night(tmp_path):
    now = datetime(2026, 9, 20, 21, 0)
    assert not registry.set_aside(registry.load_failures(tmp_path).get('c1'), now)
    assert registry.record_failure(tmp_path, 'c1', DAY_A, 'sync_estimate.py', 'boom', now) == 1
    assert not registry.set_aside(registry.load_failures(tmp_path)['c1'], now)      # one failure: try again
    assert registry.record_failure(tmp_path, 'c1', DAY_A, 'sync_estimate.py', 'boom', now) == 2
    entry = registry.load_failures(tmp_path)['c1']
    assert entry['stage'] == 'sync_estimate.py' and entry['error'] == 'boom'
    assert registry.set_aside(entry, now + timedelta(hours=1))
    assert not registry.set_aside(entry, now + timedelta(hours=registry.RETRY_AFTER_HOURS + 1))
    registry.clear_failure(tmp_path, 'c1')
    assert 'c1' not in registry.load_failures(tmp_path)


def test_keeper_goes_on_to_the_next_day_when_a_window_keeps_failing(box):
    root, windows = box
    assert keeper.next_day() == (DAY_A, 2, 3, 0)               # the queue would wait here forever
    stuck = registry.clip_key(DAY_A, windows[DAY_A][2], root / 'data' / 'raw_clips')
    registry.record_failure(root, stuck, DAY_A, 'sync_estimate.py')
    assert keeper.next_day()[0] == DAY_A                       # one failure: still worth another try
    registry.record_failure(root, stuck, DAY_A, 'sync_estimate.py')
    assert keeper.next_day() == (DAY_B, 0, 2, 0)
    assert keeper.coverage(DAY_A) == (2, 3, 1)                 # strict count is unchanged: never labelled


def test_each_worker_moves_on_by_itself_when_its_half_of_the_day_is_done(box):
    root, _ = box
    am, pm = '00:00-14:30', '14:30-23:59'
    assert keeper.next_day(am)[0] == DAY_B                     # A's morning is finished, its evening is not
    assert keeper.next_day(pm)[0] == DAY_A
    assert keeper.coverage(DAY_A, am) == (1, 1, 0)


def test_finished_pass_frees_its_slot_at_once_but_a_silent_one_still_counts_as_alive(box, monkeypatch):
    root, _ = box
    logs = root / 'data' / 'logs'
    logs.mkdir(parents=True)
    monkeypatch.setattr(keeper, 'LOGS', str(logs))
    hb = logs / 'autolabel_am.heartbeat'
    assert not keeper.pass_alive('_am')                        # no heartbeat
    hb.write_text('4242 21:00:32 start')
    monkeypatch.setattr(keeper, 'pid_alive', lambda pid: pid == 4242)
    assert keeper.pass_alive('_am')
    monkeypatch.setattr(keeper, 'pid_alive', lambda pid: False)
    assert not keeper.pass_alive('_am')                        # it ended: do not wait 45 minutes
    hb.write_text('garbled')
    assert keeper.pass_alive('_am')                            # cannot tell: keep the patient rule
    old = os.path.getmtime(hb) - 3 * 3600
    os.utime(hb, (old, old))
    assert not keeper.pass_alive('_am')                        # silent for hours: stale either way


@pytest.mark.parametrize('hour,minute,fits', [(21, 0, True), (23, 59, True), (0, 10, True),
                                              (4, 10, True), (4, 20, False), (4, 59, False)])
def test_no_pass_is_started_that_could_not_finish_a_window_before_training(hour, minute, fits):
    assert keeper.window_fits(datetime(2026, 9, 21, hour, minute)) is fits


def test_empty_window_no_longer_crashes_the_floor_projection():
    """OpenCV returns None for zero points; that was 'NoneType has no attribute reshape'."""
    cam = person3d.Camera.__new__(person3d.Camera)
    cam.K, cam.dist, cam.R, cam.C = np.eye(3), np.zeros(5), np.eye(3), np.array([0, 0, 5.0])
    empty = np.zeros((0, 2))
    assert cam.rays(empty).shape == (0, 3) and cam.on_plane(empty).shape == (0, 2)
    assert cam.height(empty, empty).shape == (0,)
    assert cam.rays(np.array([[0.1, 0.2]])).shape == (1, 3)   # ordinary input is unchanged


@pytest.fixture
def worker(tmp_path, monkeypatch):
    """auto_label with its steps faked: one window whose synchronisation fails."""
    calls = []
    monkeypatch.setattr(auto_label, 'ROOT', str(tmp_path))
    monkeypatch.setattr(auto_label, 'HEARTBEAT', str(tmp_path / 'hb'))
    monkeypatch.setattr(auto_label, 'MARGIN', 0)
    monkeypatch.setattr(auto_label, 'windows', lambda day, minutes: [hm(DAY_A, 19, 0)])
    monkeypatch.setattr(auto_label, 'door_events', lambda day: [])
    monkeypatch.setattr(sys, 'argv', ['auto_label.py', DAY_A, '10', '23:59'])
    monkeypatch.delenv('RA_RANGE', raising=False)
    fail = {'sync_estimate.py'}

    def run(args, timeout=None):
        calls.append(args[0])
        if args[0] in fail:
            auto_label.LAST_ERROR[0] = 'AttributeError: NoneType'
            return 1
        return 0
    monkeypatch.setattr(auto_label, 'run', run)
    return calls, fail, tmp_path


def test_a_window_that_fails_twice_is_left_alone_until_tomorrow(worker):
    calls, fail, root = worker
    clip = registry.clip_key(DAY_A, hm(DAY_A, 19, 0), root / 'data' / 'raw_clips')
    auto_label.main()
    assert calls[-1] == 'sync_estimate.py' and 'run_clip.py' not in calls
    entry = registry.load_failures(root)[clip]
    assert entry['attempts'] == 1 and entry['stage'] == 'sync_estimate.py' and 'NoneType' in entry['error']
    auto_label.main()
    assert registry.load_failures(root)[clip]['attempts'] == 2
    calls.clear()
    auto_label.main()                                          # third time: set aside, nothing is run
    assert calls == []
    assert registry.load_failures(root)[clip]['attempts'] == 2


def test_a_window_that_finally_works_leaves_the_failure_book(worker):
    calls, fail, root = worker
    clip = registry.clip_key(DAY_A, hm(DAY_A, 19, 0), root / 'data' / 'raw_clips')
    auto_label.main()
    assert clip in registry.load_failures(root)
    fail.clear()
    auto_label.main()
    assert 'run_clip.py' in calls and 'group_pieces.py' in calls
    assert clip not in registry.load_failures(root)
