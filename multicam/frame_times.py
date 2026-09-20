"""True frame times, taken from the recording instead of counted off the frame index.

The recorder remuxes RTSP packets straight to disk, so when the camera fails to deliver a
frame nothing is written: the file simply holds fewer frames, while its timestamps still
span the segment's full 900 s. Measured on 17.09 -- cam2's 10:30 segment holds 22253
frames and 899.4 s of timestamps, cam1's holds 21994 and the same 899.4 s. Counting 25
frames to the second therefore falls behind the recording by however much that camera
dropped, so the real moment is later than the pipeline believes: frame 9900 of that cam2
segment sits at 400.14 s while the count says 396.0, and inside one ten-minute clip the
gap reaches 14 s. That is larger than the camera-to-camera offset synchronisation is
trying to measure, and each camera drops its own frames, so it cannot cancel out.

Seeking and reading are deliberately left exactly as they were: the k-th frame read is
still the k-th frame the detector saw, so masks, crops, piece indices and the video editor
all keep their meaning. Only the time attached to that frame changes.

Timestamps are read with grab() alone -- no colour conversion -- at about 1770 frames/s,
so a 15-minute segment costs some 13 s once and is then cached.
"""
import hashlib
import os
from datetime import datetime, timedelta
from pathlib import Path

import cv2
import numpy as np

from rawsource import FPS, SEG, segments

ROOT = Path(__file__).resolve().parent
CACHE = ROOT / 'data' / 'frame_times'
REFERENCE = datetime(2000, 1, 1)   # fixed naive origin: every time here is a difference


def _cache_path(path, cache):
    """Keyed by size and mtime as well as name: a re-recorded segment is a different file."""
    stat = os.stat(path)
    key = '%s|%d|%d' % (os.path.abspath(path), stat.st_size, stat.st_mtime_ns)
    return Path(cache) / (hashlib.sha256(key.encode()).hexdigest()[:20] + '.npy')


def segment_offsets(path, cache=CACHE):
    """Seconds from the segment's own start, one per frame index, read from the container."""
    target = _cache_path(path, cache)
    if target.exists():
        return np.load(target)
    capture = cv2.VideoCapture(str(path))
    times = []
    try:
        while capture.grab():
            times.append(capture.get(cv2.CAP_PROP_POS_MSEC) / 1000.0)
    finally:
        capture.release()
    out = np.asarray(times, np.float64)
    if len(out) and not np.all(np.diff(out) >= 0):
        raise RuntimeError('Timestamps are not in order: %s' % path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + '.%d.tmp.npy' % os.getpid())
    np.save(temporary, out)
    os.replace(temporary, target)
    return out


def clip_offsets(day, cam, start, count, cache=CACHE):
    """Absolute seconds for the frames a Stream reads from `start`, or None if unavailable.

    Replays exactly what Stream.seek and Stream.read do -- open the segment covering
    `start` at the index the nominal 25 fps grid points at, then run on into the following
    segments -- so index k here is the same picture as index k there.
    """
    found = segments(cam, day)
    for position, (path, segment_start) in enumerate(found):
        if segment_start <= start < segment_start + timedelta(seconds=SEG):
            break
    else:
        return None
    index = int(round((start - segment_start).total_seconds() * FPS))
    out = np.empty(int(count), np.float64)
    filled = 0
    while filled < count and position < len(found):
        path, segment_start = found[position]
        times = segment_offsets(path, cache)
        take = min(int(count) - filled, max(0, len(times) - index))
        if take:
            out[filled:filled + take] = (segment_start - REFERENCE).total_seconds() + times[index:index + take]
            filled += take
        position += 1
        index = 0
    return out[:filled] if filled else None


def detection_times(meta, detections, cache=CACHE):
    """{cam: true time per detection}, or None if any camera's recording cannot be read.

    Column 0 of a detection row is the frame index divided by 25; that is what every frame
    lookup downstream relies on, so it is read, never written.
    """
    start = datetime.fromisoformat(meta['start'])
    out = {}
    for cam, rows in detections.items():
        frames = np.rint(np.asarray(rows)[:, 0] * FPS).astype(int) if len(rows) else np.zeros(0, int)
        wanted = int(frames.max()) + 1 if len(frames) else 1
        table = clip_offsets(meta['day'], cam, start, wanted, cache)
        if table is None or len(table) < wanted:
            return None
        out[cam] = table[frames] if len(frames) else np.zeros(0)
    return out


def drift(meta, cam, count, cache=CACHE):
    """Seconds by which the real moment is later than the counted one, at the end of the
    clip and at its worst. Dropping a frame writes nothing, so counting 25 a second falls
    behind the recording. Reported for measurement; nothing decides anything on it."""
    table = clip_offsets(meta['day'], cam, datetime.fromisoformat(meta['start']), count, cache)
    if table is None or not len(table):
        return None
    behind = (table - table[0]) - np.arange(len(table)) / FPS
    return float(behind[-1]), float(np.abs(behind).max())
