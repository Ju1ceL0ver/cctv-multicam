"""Learned scores must never bypass physical limits or create predicted masks."""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'multicam'))
from door_seg_link_model import LearnedTracker, features


def observation(x):
    mask=np.zeros((184,320),bool)
    mask[25:50,x//4:x//4+10]=True
    return mask,np.array([x,100,x+40,200]),.9,np.zeros(2),np.zeros(2)


class ConfidentModel:
    def predict_proba(self, x):
        return np.tile([.01,.99],(len(x),1))


def test_classifier_cannot_override_maximum_speed():
    tr=LearnedTracker(ConfidentModel(),threshold=.7)
    assert tr.update(0,[observation(100)])[0][0]==1
    assert tr.update(1,[observation(400)])[0][0]==2


def test_short_recovery_then_expiry_without_synthetic_observations():
    tr=LearnedTracker(ConfidentModel(),threshold=.7)
    tr.update(0,[observation(100)])
    assert tr.update(1,[])==[]
    assert tr.update(6,[observation(100)])[0][0]==1
    assert tr.update(13,[observation(100)])[0][0]==2


def test_features_use_observations_without_owner_labels():
    a=observation(100)
    f=features(a,a,1)
    assert f.shape==(11,)
    assert np.isfinite(f).all()
    assert f[4]==f[5]==1
    assert f[9]==.5  # absent appearance descriptor has a neutral value
