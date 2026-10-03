"""The rough drafts as v2 targets, the stand pasted in; the mapped mask store; the decoder cap."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'multicam'))
import v2_data as VD            # noqa: E402
import sam31_reid               # noqa: E402


def label_with(people):
    lab = np.zeros((720, 1280), np.uint8)
    for v, (x1, y1, x2, y2) in enumerate(people, 1):
        lab[y1:y2, x1:x2] = v
    return lab


def test_draft_targets_boxes_and_grid():
    tg = VD.draft_targets(label_with([(100, 100, 200, 400), (600, 300, 640, 360)]), 'cam1')
    assert tg['masks'].shape == (2, VD.GH, VD.GW)
    x1, y1 = 100 * VD.FW / 1280, 100 * VD.FH / 720
    cx, cy, w, h = tg['boxes'][0]
    assert abs((cx - w / 2) * VD.FW - x1) <= 8 and abs((cy - h / 2) * VD.PH - y1) <= 8
    assert list(tg['person']) == [0, 1] and (tg['zone'] == -1).all() and not tg['has_reid'].any()
    assert tg['masks'][:, VD.FH // 4:].sum() == 0                 # the padding rows stay empty


def test_the_stand_is_added_where_nobody_covers_it():
    poster = np.zeros((VD.GH, VD.GW), bool)
    poster[20:60, 400:420] = True
    tg = VD.draft_targets(label_with([(100, 100, 200, 400)]), 'cam1', poster)
    assert len(tg['masks']) == 2 and (tg['masks'][1] == poster).all()
    # somebody standing in front: the stand is what is left of it
    x1, x2 = int(400 * 4 * 1280 / VD.FW), int(410 * 4 * 1280 / VD.FW)
    tg = VD.draft_targets(label_with([(x1, 0, x2, 400)]), 'cam1', poster)
    assert len(tg['masks']) == 2 and tg['masks'][1].sum() < poster.sum() and not (tg['masks'][1] & tg['masks'][0]).any()
    # the draft already has it as a person: not added twice
    y1, y2 = int(20 * 4 * 720 / VD.FH), int(60 * 4 * 720 / VD.FH)
    tg = VD.draft_targets(label_with([(int(400 * 4 * 1280 / VD.FW), y1, int(420 * 4 * 1280 / VD.FW), y2)]), 'cam1', poster)
    assert len(tg['masks']) == 1


def test_masks_mapped_equal_loaded(tmp_path):
    rows = np.array([[0, 0, 0, 1, 10, 10, 14, 13]], np.float64)
    m = np.zeros((3, 4), bool); m[1, 2] = m[0, 0] = True
    bits = np.packbits(m)
    np.savez_compressed(tmp_path / 'chunks.npz', rows=rows, buf=bits, offs=np.array([0, len(bits)], np.int64))
    a = sam31_reid.Masks(tmp_path / 'chunks.npz')
    sam31_reid.Masks.to_npy(tmp_path / 'chunks.npz')
    b = sam31_reid.Masks(tmp_path / 'chunks.npz', mmap=True)
    assert isinstance(b.buf, np.memmap)
    assert (a.crop(0) == b.crop(0)).all() and (a.crop(0) == m).all()


def test_at_most_max_open_decoders():
    class Fake:
        def __init__(self):
            self.released = False

        def release(self):
            self.released = True

        def set(self, *a):
            return True

        def read(self):
            return False, None
    ws = []
    VD._OPEN.clear()
    for k in range(VD.MAX_OPEN + 2):
        w = VD.Window.__new__(VD.Window)
        w.cap, w.pos, w.video = Fake(), 0, 'none'
        ws.append(w)
        w.frame(0)
    assert len(VD._OPEN) == VD.MAX_OPEN
    assert ws[0].cap is None and ws[1].cap is None and ws[-1].cap is not None
