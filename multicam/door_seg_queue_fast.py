"""Experimental side queue cache; no automatic production integration.

Queued RGB frames must be immutable, as in door_live's window reader.
Keep the actual frame reference to avoid a stale hit after Python reuses an id.
The lock gives one preparation per frame, even when several people request it.
"""
from collections import OrderedDict
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
                return old[1]
            result = self.prepare(frame)
            self.frames[key] = (frame, result)
            self.misses += 1
            while len(self.frames) > self.keep:
                self.frames.popitem(last=False)
            return result


class Live(BaseLive):
    def __init__(self):
        super().__init__()
        self.prepared_frames = PreparedFrames(self.clf._small)
        self.clf._small = self.prepared_frames
