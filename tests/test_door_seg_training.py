"""Resume safety and the owner's plateau rule (no GPU or Ultralytics needed)."""
import sys
from pathlib import Path

import pytest
import torch
import datetime

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'multicam'))
from door_seg import checkpoint_finite, plateau
from door_seg_supervise import deadline, night


def test_rejects_poisoned_model_or_optimizer():
    checkpoint_finite({'model': {'yolo': {'x': torch.ones(3)}}, 'opt': {'state': {0: {'step': torch.tensor(3)}}}})
    with pytest.raises(ValueError, match='model.yolo.x'):
        checkpoint_finite({'model': {'yolo': {'x': torch.tensor(float('nan'))}}})
    with pytest.raises(ValueError, match='opt.state.0.exp_avg'):
        checkpoint_finite({'model': {}, 'opt': {'state': {0: {'exp_avg': torch.tensor(float('inf'))}}}})


def test_plateau_needs_three_spaced_checks():
    assert not plateau([(2604, .9319), (4604, .9330), (6604, .9325)])
    assert plateau([(2604, .9319), (4604, .9330), (6604, .9325), (8604, .9340)])
    assert not plateau([(2604, .9319), (2704, .932), (2804, .931), (2904, .931)])


def test_improvement_resets_patience_and_nan_is_not_a_check():
    assert not plateau([(0, .90), (2000, .901), (4000, .902), (6000, .91), (8000, .912)])
    assert not plateau([(0, .90), (2000, float('nan')), (4000, .90), (6000, .90)])
    assert plateau([(0, .90), (2000, .91), (4000, .912), (6000, .911), (8000, .910)])


def test_day_boundary_never_restarts_training_for_another_daytime_window():
    d = datetime.datetime(2026, 10, 10, 3)
    assert night(d)
    assert not night(d.replace(hour=9, minute=40))
    assert not night(d.replace(hour=15))
    assert night(d.replace(hour=21))
    assert deadline(d, '09:40') == d.replace(hour=9, minute=40)
    assert deadline(d.replace(hour=21), '09:40') == d.replace(day=11, hour=9, minute=40)
