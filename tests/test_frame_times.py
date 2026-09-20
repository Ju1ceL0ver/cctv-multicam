"""Frame times read from the recording, not counted off the frame index."""
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'multicam'))
import frame_times as ft                                 # noqa: E402

DAY = '20260917'
FIRST = datetime(2026, 9, 17, 10, 0, 0)


@pytest.fixture
def recording(monkeypatch, tmp_path):
    """Two 900 s segments. The first delivered every frame; the second lost 250 of them
    in one burst a third of the way in, exactly as the shop's camera does."""
    full = np.arange(22500) / 25.0
    lossy = np.concatenate([np.arange(7500) / 25.0, 310.0 + np.arange(14750) / 25.0])
    tables = {'a.mp4': full, 'b.mp4': lossy}
    monkeypatch.setattr(ft, 'segments', lambda cam, day: [
        ('a.mp4', FIRST), ('b.mp4', FIRST + timedelta(seconds=900))] if day == DAY else [])
    monkeypatch.setattr(ft, 'segment_offsets', lambda path, cache=None: tables[path])
    return tables


def test_time_comes_from_the_recording_not_the_frame_number(recording):
    out = ft.clip_offsets(DAY, 'cam1', FIRST + timedelta(seconds=900), 22250)
    base = (FIRST + timedelta(seconds=900) - ft.REFERENCE).total_seconds()
    assert out[0] == base and out[7499] == base + 299.96
    # the burst: the next frame is ten seconds later, not one 25th of a second
    assert out[7500] == pytest.approx(base + 310.0)
    counted = base + np.arange(len(out)) / 25.0
    assert (out - counted)[-1] == pytest.approx(10.0, abs=1e-6)   # the real moment is later


def test_a_clip_that_crosses_a_segment_boundary_keeps_running(recording):
    start = FIRST + timedelta(seconds=880)     # 20 s before the boundary
    out = ft.clip_offsets(DAY, 'cam1', start, 1000)
    assert len(out) == 1000
    boundary = (FIRST + timedelta(seconds=900) - ft.REFERENCE).total_seconds()
    assert out[499] == pytest.approx(boundary - 0.04)   # last frame of the first segment
    assert out[500] == pytest.approx(boundary)          # first frame of the second
    assert np.all(np.diff(out) >= 0)


def test_asking_for_more_than_was_recorded_returns_what_exists(recording):
    out = ft.clip_offsets(DAY, 'cam1', FIRST + timedelta(seconds=1780), 10_000)
    assert len(out) == 250      # the recording stops there; the rest is not invented


def test_a_day_that_was_never_recorded_has_no_times(recording):
    assert ft.clip_offsets('20990101', 'cam1', FIRST, 10) is None
    assert ft.clip_offsets(DAY, 'cam1', FIRST - timedelta(days=1), 10) is None


def test_detection_times_are_read_off_column_zero_and_never_write_it(recording):
    meta = {'day': DAY, 'start': (FIRST + timedelta(seconds=900)).isoformat()}
    rows = np.zeros((3, 10), np.float32)
    rows[:, 0] = [0, 7499 / 25.0, 7500 / 25.0]
    detections = {'cam1': rows, 'cam2': np.zeros((0, 10), np.float32)}
    before = rows.copy()
    times = ft.detection_times(meta, detections)
    assert np.array_equal(rows, before)
    assert times['cam1'][1] == pytest.approx(times['cam1'][0] + 299.96)
    assert times['cam1'][2] == pytest.approx(times['cam1'][0] + 310.0)
    assert len(times['cam2']) == 0


def test_detection_times_give_up_rather_than_guess(recording, monkeypatch):
    meta = {'day': '20990101', 'start': FIRST.isoformat()}
    rows = np.zeros((1, 10), np.float32)
    assert ft.detection_times(meta, {'cam1': rows}) is None


def test_drift_reports_how_far_the_counted_time_ran_ahead(recording):
    meta = {'day': DAY, 'start': (FIRST + timedelta(seconds=900)).isoformat()}
    last, largest = ft.drift(meta, 'cam1', 15000)
    assert last == pytest.approx(10.0, abs=1e-6) and largest == pytest.approx(10.0, abs=1e-6)


def test_timestamps_are_read_once_and_then_cached(tmp_path, monkeypatch):
    """A segment costs 13 s of grabbing; every clip that touches it must not pay again."""
    opened = []

    class Capture:
        def __init__(self, path):
            opened.append(path)
            self.i = -1
        def grab(self):
            self.i += 1
            return self.i < 5
        def get(self, prop):
            return self.i * 40.0
        def release(self):
            pass
    monkeypatch.setattr(ft.cv2, 'VideoCapture', Capture)
    video = tmp_path / 'seg.mp4'
    video.write_bytes(b'not really a video')
    first = ft.segment_offsets(str(video), tmp_path / 'cache')
    again = ft.segment_offsets(str(video), tmp_path / 'cache')
    assert np.array_equal(first, [0, .04, .08, .12, .16]) and np.array_equal(first, again)
    assert len(opened) == 1
    video.write_bytes(b'a different recording entirely')
    ft.segment_offsets(str(video), tmp_path / 'cache')
    assert len(opened) == 2      # same name, new file: not the same timestamps


def test_timestamps_that_go_backwards_are_refused(tmp_path, monkeypatch):
    class Capture:
        def __init__(self, path): self.i = -1
        def grab(self):
            self.i += 1
            return self.i < 3
        def get(self, prop): return [0.0, 80.0, 40.0][self.i]
        def release(self): pass
    monkeypatch.setattr(ft.cv2, 'VideoCapture', Capture)
    video = tmp_path / 'seg.mp4'
    video.write_bytes(b'x')
    with pytest.raises(RuntimeError, match='not in order'):
        ft.segment_offsets(str(video), tmp_path / 'cache')
