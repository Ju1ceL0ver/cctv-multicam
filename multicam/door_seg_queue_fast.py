"""Experimental side queue cache; no automatic production integration.

Queued RGB frames must be immutable, as in door_live's window reader.
Keep the actual frame reference to avoid a stale hit after Python reuses an id.
The lock gives one preparation per frame, even when several people request it.
"""
from collections import OrderedDict
from concurrent.futures import Future
import threading

from door_combo import Live as BaseLive


class PreparedFrames:
    def __init__(self, prepare, keep=8):
        self.prepare, self.keep = prepare, keep
        self.frames = OrderedDict()
        self.lock = threading.Lock()
        self.hits = self.misses = 0

    def __call__(self, frame):
        with self.lock:
            key = id(frame)
            old = self.frames.get(key)
            if old is not None and old[0] is frame:
                self.frames.move_to_end(key)
                self.hits += 1
                future, owner = old[1], False
            else:
                future, owner = Future(), True
                self.frames[key] = (frame, future)
                self.misses += 1
                while len(self.frames) > self.keep:
                    self.frames.popitem(last=False)
        # Different frames can prepare concurrently; only requests for the same
        # frame wait for its future. Never hold the global lock across resize.
        if owner:
            try:
                future.set_result(self.prepare(frame))
            except BaseException as exc:
                future.set_exception(exc)
        return future.result()


def prepare_rgb(frame):
    import cv2
    import numpy as np
    # Window frames are immutable RGB arrays already at 1280x720. Identity
    # resize merely copied 2.6MiB for every distinct frame.
    rgb = frame if frame.shape[:2] == (720,1280) else cv2.resize(frame,(1280,720),interpolation=cv2.INTER_LINEAR)
    small = cv2.resize(rgb.astype(np.float32),(320,180),interpolation=cv2.INTER_AREA).astype(np.uint8)
    return rgb, small


class Live(BaseLive):
    def __init__(self):
        super().__init__()
        self.prepared_frames = PreparedFrames(prepare_rgb)
        self.clf._small = self.prepared_frames
