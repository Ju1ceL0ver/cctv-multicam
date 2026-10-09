"""Identity safety, displacement units, missing frames and export compatibility."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'multicam'))
from door_seg_track import Tracker, shifted_iou
from door_seg_run import contexts, save_chunks


def pred(x, dx=0, backward=None):
    m = np.zeros((184, 320), bool)
    m[25:50, int(x / 4):int(x / 4) + 10] = True
    return (m, np.array([x, 100, x + 40, 200], float), 0.9,
            np.array([-dx if backward is None else backward, 0.0]), np.array([dx, 0.0]))


def ids(tracker, tick, ps):
    return [pid for pid, _ in tracker.update(tick, ps)]


def test_offsets_follow_people_when_prediction_order_changes():
    tr = Tracker()
    assert ids(tr, 0, [pred(100, 16), pred(180, -16)]) == [1, 2]
    assert ids(tr, 1, [pred(164, -16), pred(116, 16)]) == [2, 1]
    assert ids(tr, 2, [pred(132, 16), pred(148, -16)]) == [1, 2]
    assert ids(tr, 3, [pred(132, -16), pred(148, 16)]) == [2, 1]


def test_two_detections_cannot_take_one_number():
    tr = Tracker()
    ids(tr, 0, [pred(100)])
    got = ids(tr, 1, [pred(100), pred(100)])
    assert len(set(got)) == 2
    assert 1 not in got  # exact tie: identity is uncertain


def test_forward_agreement_does_not_override_bad_backward_head():
    tr = Tracker()
    ids(tr, 0, [pred(100, 8)])
    assert ids(tr, 1, [pred(108, 8, backward=80)]) == [2]


def test_short_gap_recovered_without_hallucinated_observations():
    tr = Tracker()
    ids(tr, 0, [pred(100, 8)])
    assert ids(tr, 1, [pred(108, 8)]) == [1]
    assert tr.update(2, []) == []
    assert ids(tr, 3, [pred(124, 8)]) == [1]
    assert tr.stats['recovered'] == 1
    assert ids(tr, 29, [pred(124)]) == [2]


def test_large_jump_and_duplicate_tick_rejected():
    tr = Tracker()
    ids(tr, 0, [pred(100)])
    assert ids(tr, 1, [pred(300)]) == [2]
    with pytest.raises(ValueError):
        ids(tr, 1, [])


def test_translated_mask_uses_input_pixels_without_wrap():
    assert shifted_iou(pred(100)[0], pred(116)[0], [16, 0]) == 1.0
    assert shifted_iou(pred(100)[0], pred(116)[0], [1280, 0]) == 0.0


class Cap:
    def __init__(self, n):
        self.it = iter(range(n))

    def read(self):
        try:
            return True, next(self.it)
        except StopIteration:
            return False, None


@pytest.mark.parametrize('n', [0, 1, 3, 20])
def test_temporal_context_has_every_tick_and_clamps_only_edges(n):
    got = list(contexts(Cap(n)))
    assert len(got) == n
    for tick, frames in got:
        assert frames == {o: max(0, min(n - 1, tick + o)) for o in range(-5, 6)}


def test_export_roundtrip_crop_and_person_number(tmp_path):
    pytest.importorskip('cv2')
    from sam31_reid import Masks, link_seams
    save_chunks(tmp_path, [(7, 42, pred(100))], 1280, 720)
    m = Masks(tmp_path / 'chunks.npz')
    assert m.rows.shape == (1, 8)
    assert list(m.rows[0, :4]) == [0, 7, 42, 0.9]
    assert m.crop(0).shape == (100, 40)
    assert m.crop(0).all()
    owned, _ = link_seams(m, {0: 0})
    assert owned == {0: [0]}  # reports must key on zero-based pieces
    save_chunks(tmp_path, [], 1280, 720)
    assert Masks(tmp_path / 'chunks.npz').rows.shape == (0, 8)
