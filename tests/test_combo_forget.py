"""09.10: the live door forgets the side observations of a window whose folder is gone (door_combo.Live.forget)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'multicam'))


def test_forget_drops_only_that_window():
    import door_combo
    live = door_combo.Live.__new__(door_combo.Live)          # no models: only the stores
    live.raw = {('w1', 0, 0): 1, ('w1', 1, 0): 2, ('w2', 0, 0): 3}
    live.obs = {('w1', 0, 0): 1, ('w2', 0, 0): 3}
    live.pending = {('w1', 1, 0): 5}
    live.forget('w1')
    assert list(live.raw) == [('w2', 0, 0)] and list(live.obs) == [('w2', 0, 0)] and live.pending == {}
    live.forget('w9')                                          # nothing of it: nothing happens
    assert list(live.raw) == [('w2', 0, 0)]
