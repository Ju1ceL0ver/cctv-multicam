"""Fast path must preserve channels, cropped masks and displacement units."""
import sys
from pathlib import Path
import numpy as np
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'multicam'))
import door_seg as S
from door_seg_fast import Channels,previous_union,predict


def test_channels_exact_and_cache_bounded():
    rng=np.random.default_rng(1)
    bg=rng.integers(0,256,(S.H,S.W,3),dtype=np.uint8)
    frames={k:rng.integers(0,256,(S.H,S.W,3),dtype=np.uint8) for k in range(-5,6)}
    prev=rng.integers(0,2,(S.H,S.W),dtype=np.uint8)
    fast=Channels(bg)
    assert np.array_equal(fast(frames,prev),S.channels(frames,bg,prev))
    frames={k:frames[0] for k in range(-5,6)}
    assert np.array_equal(fast(frames,prev),S.channels(frames,bg,prev))
    assert len(fast.grey)==1


def test_previous_mask_union_matches_individual_resizes():
    import cv2
    rng=np.random.default_rng(2)
    masks=[rng.integers(0,2,(S.MH,S.MW),dtype=np.uint8)>0 for _ in range(4)]
    original=np.zeros((S.H,S.W),np.uint8)
    for m in masks:original|=cv2.resize(m[:180].astype(np.uint8),(S.W,S.H),interpolation=cv2.INTER_NEAREST)
    assert np.array_equal(previous_union([(m,) for m in masks]),original)
    assert not previous_union([]).any()


def test_vectorized_mask_and_offsets_equal_original():
    torch=pytest.importorskip('torch')
    torch.manual_seed(1)
    det=torch.zeros((1,3,38));det[0,:,:4]=torch.tensor([[1.5,40,100.1,200.2],[110,100,150,200],[900,0,901,1]])
    det[0,:,4]=torch.tensor([.8,.7,.6]);det[0,:,6:]=torch.randn(3,32)
    proto=torch.randn(1,32,S.MH,S.MW)
    offset=torch.randn(1,4,S.GH,S.GW)
    class Fake:
        def yolo(self,img):return ((det,proto),)
        def off(self,feat):return offset
        feat=None
    img=torch.zeros(1,S.C,S.HP,S.W)
    a=S.predict(Fake(),img);b=predict(Fake(),img)
    assert len(a)==len(b)
    for aa,bb in zip(a,b):
        assert np.array_equal(aa[0],bb[0])
        assert np.array_equal(aa[1],bb[1])
        np.testing.assert_allclose(aa[3],bb[3],atol=1e-5)
        np.testing.assert_allclose(aa[4],bb[4],atol=1e-5)
