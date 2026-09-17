"""Frames of a camera-day addressed by server time, across 15-minute raw segments.

Segment N of a recording starts at its HHMMSS prefix + N * 900 s; each holds exactly
22500 frames at 25 fps (verified), so a frame's server time is segment start + index/25."""
import os, re, glob, cv2
from datetime import datetime, timedelta

RAW = r'C:\Users\ArykovAA\cctv_ai\retail_analytics\runs\raw'
FPS = 25.0
SEG = 900


def segments(cam, day):
    dirs = [os.path.join(RAW, cam, day)] + ([os.path.join(RAW, day)] if cam == 'cam1' else [])
    out = []
    for d in dirs:
        for p in sorted(glob.glob(os.path.join(d, '*.mp4'))):
            m = re.match(r'(\d{6})_(\d{4})\.mp4$', os.path.basename(p))
            if m:
                out.append((p, datetime.strptime(day + m.group(1), '%Y%m%d%H%M%S') + timedelta(seconds=SEG * int(m.group(2)))))
    return out


class Stream:
    def __init__(self, cam, day):
        self.cam, self.segs, self.cap, self.i = cam, segments(cam, day), None, -1

    def seek(self, t):
        for i, (p, st) in enumerate(self.segs):
            if st <= t < st + timedelta(seconds=SEG):
                self._open(i, int(round((t - st).total_seconds() * FPS)))
                return True
        raise ValueError('%s: no segment covers %s' % (self.cam, t))

    def _open(self, i, k):
        if self.cap is not None:
            self.cap.release()
        self.i, self.cap = i, cv2.VideoCapture(self.segs[i][0])
        if k:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, k)
        self.k = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))

    def read(self):
        while True:
            ok, fr = self.cap.read()
            if ok:
                t = self.segs[self.i][1] + timedelta(seconds=self.k / FPS)
                self.k += 1
                return t, fr
            if self.i + 1 >= len(self.segs):
                return None, None
            self._open(self.i + 1, 0)
