"""Light copies of a camera-day, for watching the whole day in a browser. CPU only.

The raw segments are 2560x1440 at ~4 Mbit/s -- 17 GB per camera for 17.09 -- far too much
to stream through a quick tunnel. Each 15-minute segment is re-encoded to 960 px with
libx264 at below-normal priority, so the production counter keeps the machine.

Two things are kept exactly:

* **Every frame, in order.** Frame k of the copy is frame k of the original, which is the
  frame the detector read (see frame_times: the k-th frame read is the k-th frame seen). So
  a detection box drawn on the copy is the box the detector found, with no resampling.
* **The original timestamps** (`-fps_mode passthrough`). The camera drops frames in bursts
  and the file keeps their time gaps; counting 25 a second falls behind by up to 14 s inside
  one window. The copy plays in the recording's own time, and the sidecar stores where the
  40 ms rhythm breaks, which is enough to turn any frame index into its true time.
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'data' / 'day_proxy'
WIDTH = 960
STEP = 0.04            # the camera's own frame interval; anything else is a drop burst
SLACK = 0.004


def paths(day, cam, raw, out=OUT):
    base = Path(out) / day / cam / Path(raw).stem
    return base.with_suffix('.mp4'), base.with_suffix('.json')


def frame_times(path):
    """Seconds from the start of the file, one per frame, read off the container. grab()
    alone does not decode colour, so a 15-minute segment costs seconds, not minutes."""
    capture = cv2.VideoCapture(str(path))
    times = []
    try:
        while capture.grab():
            times.append(capture.get(cv2.CAP_PROP_POS_MSEC) / 1000.0)
    finally:
        capture.release()
    return np.asarray(times, np.float64)


def breaks(times):
    """[[index, seconds], ...] at the first frame and wherever the 40 ms rhythm breaks.
    Between two breaks every frame is exactly 40 ms after the previous one."""
    if not len(times):
        return []
    out = [[0, round(float(times[0]), 4)]]
    steps = np.diff(times)
    for k in np.nonzero(np.abs(steps - STEP) > SLACK)[0]:
        out.append([int(k) + 1, round(float(times[k + 1]), 4)])
    return out


def expand(points, frames):
    """The inverse of breaks(): a time for every frame index."""
    times = np.empty(frames, np.float64)
    for n, (k, t) in enumerate(points):
        end = points[n + 1][0] if n + 1 < len(points) else frames
        times[k:end] = t + STEP * np.arange(end - k)
    return times


def _ffmpeg():
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()


def encode(raw, target, threads=2, ffmpeg=None):
    ffmpeg = ffmpeg or _ffmpeg()
    common = ['-an', '-vf', 'scale=%d:-2' % WIDTH, '-c:v', 'libx264', '-preset', 'veryfast',
              '-crf', '27', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', '-threads', str(threads)]
    flags = getattr(subprocess, 'BELOW_NORMAL_PRIORITY_CLASS', 0)
    for keep in (['-fps_mode', 'passthrough'], ['-vsync', '0']):      # newer / older ffmpeg
        done = subprocess.run([ffmpeg, '-y', '-loglevel', 'error', '-threads', str(threads), '-i', str(raw)]
                              + keep + common + [str(target)], capture_output=True, text=True,
                              creationflags=flags)
        if done.returncode == 0:
            return
    raise RuntimeError('ffmpeg: ' + done.stderr[-400:])


def make(day, cam, raw, start, out=OUT, threads=2, ffmpeg=None):
    """One segment. Refuses a copy whose frames do not line up with the original's."""
    mp4, side = paths(day, cam, raw, out)
    if mp4.exists() and side.exists():
        return json.loads(side.read_text())
    mp4.parent.mkdir(parents=True, exist_ok=True)
    temporary = mp4.with_name(mp4.stem + '.tmp.mp4')
    began = time.time()
    encode(raw, temporary, threads, ffmpeg)
    encoded = time.time() - began
    original, copy = frame_times(raw), frame_times(temporary)
    timed = time.time() - began - encoded
    if len(copy) != len(original):
        temporary.unlink(missing_ok=True)
        raise RuntimeError('%s: copy has %d frames, original %d' % (raw, len(copy), len(original)))
    shift = float(copy[0] - original[0]) if len(copy) else 0.0
    worst = float(np.abs(copy - original - shift).max()) if len(copy) else 0.0
    if worst > 0.021:
        temporary.unlink(missing_ok=True)
        raise RuntimeError('%s: copy timestamps drift %.3f s from the original' % (raw, worst))
    height = int(cv2.VideoCapture(str(temporary)).get(cv2.CAP_PROP_FRAME_HEIGHT))
    stat = os.stat(raw)
    record = {'day': day, 'cam': cam, 'raw': Path(raw).name, 'start': start.isoformat(),
              'frames': int(len(original)), 'breaks': breaks(original), 'shift': round(shift, 4),
              'duration': round(float(original[-1] - original[0]) if len(original) else 0.0, 3),
              'width': WIDTH, 'height': height, 'bytes': 0,
              'source': {'bytes': stat.st_size, 'mtime_ns': stat.st_mtime_ns},
              'encode_s': round(encoded, 1), 'timing_s': round(timed, 1)}
    os.replace(temporary, mp4)
    record['bytes'] = mp4.stat().st_size
    side.write_text(json.dumps(record))
    return record


def day_segments(day):
    """Both cameras, interleaved in time, so the morning of both is ready first."""
    from rawsource import segments
    rows = [(start, cam, raw) for cam in ('cam1', 'cam2') for raw, start in segments(cam, day)]
    return sorted(rows)


if __name__ == '__main__':
    day = sys.argv[1]
    limit = int(sys.argv[2]) if len(sys.argv) > 2 else None
    only = sys.argv[3] if len(sys.argv) > 3 else None
    done = 0
    for start, cam, raw in day_segments(day):
        if only and cam != only:
            continue
        try:
            record = make(day, cam, raw, start)
            print(json.dumps({k: record[k] for k in ('cam', 'raw', 'frames', 'bytes', 'encode_s', 'timing_s')}),
                  flush=True)
        except Exception as exc:                                   # one bad segment must not stop the day
            # A recording the machine never finished writing (a reboot mid-segment) is not
            # "still encoding": say so in the sidecar, so the player can tell the owner.
            _, side = paths(day, cam, raw)
            side.parent.mkdir(parents=True, exist_ok=True)
            side.write_text(json.dumps({'day': day, 'cam': cam, 'raw': Path(raw).name,
                                        'start': start.isoformat(), 'frames': 0,
                                        'error': str(exc)[:300]}))
            print(json.dumps({'cam': cam, 'raw': Path(raw).name, 'error': str(exc)}), flush=True)
        done += 1
        if limit and done >= limit:
            break
