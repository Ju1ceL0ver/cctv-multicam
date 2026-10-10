"""Memory isolation, BPTT, conservative identity labels, and ordered unroll."""
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'multicam'))
from door_seg_temporal import TemporalHeads, pool_instances, pool_prediction_masks, identity_loss
import door_seg_sequence as Q

torch.set_num_threads(2)


def heads():
    torch.manual_seed(3)
    return TemporalHeads(8, hidden=4, embedding=6)


def test_new_memory_preserves_original_features_exactly():
    model = heads()
    x = torch.randn(2, 8, 5, 7)
    enhanced, embedding, state = model(x, ['cam1', 'cam2'], [10, 50])
    assert torch.equal(x, enhanced)
    assert embedding.shape == (2, 6, 5, 7)
    assert state.hidden.grad_fn is None


def test_reset_is_per_camera_and_gap_not_whole_batch():
    model = heads()
    x = torch.randn(2, 8, 6, 8)
    _, _, old = model(x, ['a', 'b'], [1, 1])
    _, _, continued = model(x, ['a', 'b'], [2, 2], old)
    _, _, reset = model(x, ['new', 'b'], [2, 2], old)
    _, _, fresh = model(x, ['new', 'b'], [2, 2])
    assert torch.equal(reset.hidden[0], fresh.hidden[0])
    assert torch.equal(reset.hidden[1], continued.hidden[1])
    _, _, gap = model(x, ['a', 'b'], [3, 3], old)
    assert torch.equal(gap.hidden, fresh.hidden)
    old.hidden[0].fill_(float('nan'))
    _, _, reset_bad = model(x, ['new', 'b'], [2, 2], old)
    assert torch.isfinite(reset_bad.hidden).all()


def test_shape_change_resets_and_inference_does_not_retain_graph():
    model = heads()
    _, _, state = model(torch.randn(1, 8, 4, 4), ['a'], [0])
    x = torch.randn(2, 8, 6, 8)
    _, _, changed = model(x, ['a', 'b'], [1, 1], state)
    _, _, fresh = model(x, ['a', 'b'], [1, 1])
    assert torch.equal(changed.hidden, fresh.hidden)
    for tick in range(2, 10):
        _, _, changed = model(x, ['a', 'b'], [tick, tick], changed)
        assert changed.hidden.grad_fn is None


def test_bptt_reaches_previous_frames_but_not_other_stream():
    model = heads()
    nn.init.normal_(model.residual.weight, std=0.1)
    past = torch.randn(1, 8, 4, 4, requires_grad=True)
    present = torch.randn_like(past, requires_grad=True)
    _, _, state = model(past, ['a'], [0], retain_graph=True)
    enhanced, _, _ = model(present, ['a'], [1], state, retain_graph=True)
    enhanced.square().mean().backward()
    assert past.grad.abs().sum() > 0
    past2 = past.detach().requires_grad_()
    _, _, state2 = model(past2, ['a'], [0], retain_graph=True)
    enhanced2, _, _ = model(present.detach(), ['b'], [1], state2, retain_graph=True)
    enhanced2.square().mean().backward()
    assert past2.grad is None or past2.grad.abs().sum() == 0


def test_pooling_excludes_background_and_empty_ids_and_has_gradients():
    dense = torch.tensor([[[[1., 0.], [9., 0.]], [[0., 2.], [9., 2.]]]], requires_grad=True)
    masks = torch.tensor([[[1, 2], [0, 2]]])
    vectors, lanes, ids = pool_instances(dense, masks)
    assert ids.tolist() == [1, 2] and lanes.tolist() == [0, 0]
    assert torch.allclose(vectors, torch.eye(2))
    vectors.sum().backward()
    assert dense.grad[0, :, 1, 0].sum() == 0
    empty, _, _ = pool_instances(dense, torch.zeros_like(masks))
    (empty.sum() * 0).backward()
    assert empty.shape == (0, 2)


def test_prediction_pooling_keeps_overlap_and_marks_empty():
    dense = torch.ones(1, 4, 3, 3)
    masks = torch.stack([torch.ones(6, 6), torch.ones(6, 6), torch.zeros(6, 6)])
    vectors, valid = pool_prediction_masks(dense, masks)
    assert valid.tolist() == [True, True, False]
    assert torch.equal(vectors[0], vectors[1])
    assert torch.equal(vectors[2], torch.zeros(4))


def test_identity_prefers_correct_person_and_ignores_other_namespaces():
    good = torch.tensor([[1., 0.], [0., 1.], [1., 0.], [0., 1.]], requires_grad=True)
    ids, namespaces, ticks = [0, 1, 0, 1], [0] * 4, [0, 0, 1, 1]
    good_loss, anchors = identity_loss(good, ids, namespaces, ticks)
    bad = torch.tensor([[1., 0.], [0., 1.], [0., 1.], [1., 0.]])
    bad_loss, _ = identity_loss(bad, ids, namespaces, ticks)
    assert anchors == 4 and good_loss < bad_loss
    good_loss.backward()
    assert torch.isfinite(good.grad).all()
    extended = torch.cat((good.detach(), torch.randn(4, 2)))
    extended_loss, extended_anchors = identity_loss(extended, ids + [0, 1, 2, 3], namespaces + [1] * 4, ticks + [2]*4)
    assert extended_anchors == 4 and torch.allclose(good_loss, extended_loss)


@pytest.mark.parametrize('ids,ns,ticks', [([-1,-1], [0,0], [0,1]),
                                       ([0,0], [0,1], [0,1]),
                                       ([0,1], [0,0], [0,1]),
                                       ([0,0], [0,0], [0,0])])
def test_unusable_identity_pairs_produce_finite_connected_zero(ids, ns, ticks):
    x = torch.randn(2, 5, requires_grad=True)
    loss, anchors = identity_loss(x, ids, ns, ticks)
    assert loss == 0 and anchors == 0
    loss.backward()
    assert torch.isfinite(x.grad).all()


def test_sequences_reject_session_gaps_and_test_manifest(tmp_path):
    source = SimpleNamespace(d=tmp_path, starts=[(1,1),(1,2),(2,3),(2,4),(2,6)])
    assert Q.sequence_starts(source, 2) == [(1,1),(2,3)]
    with pytest.raises(ValueError, match='TRAIN'):
        Q.sample_sequence(source, 2, allowed_sources=[])
    with pytest.raises(ValueError, match='gap'):
        Q.sample_sequence(source, 2, at=(1,2), allowed_sources=[tmp_path])


def test_sequence_sampling_keeps_identity_order_and_shared_brightness(tmp_path, monkeypatch):
    reads = []
    def read(a, b):
        reads.append((a, b))
        return {k: None for k in range(a, b)}
    source = SimpleNamespace(d=tmp_path, starts=[(1,10),(1,11)], read=read,
                             people=lambda s,k: {70: (np.ones((2,2)), None), 20: (np.ones((1,1)), None)})
    monkeypatch.setattr(Q.S, 'H', 2)
    monkeypatch.setattr(Q.S, 'spoil', lambda masks,rng: np.ones((2,2), np.uint8))
    def sample(src,rng,train,at,cache):
        assert not train and at[1] in cache
        return {'img':np.full((17,2,2), 100, np.uint8), 'meta':{'tick':at[1]}}
    monkeypatch.setattr(Q.S, 'sample', sample)
    seq = Q.sample_sequence(source, 2, allowed_sources=[tmp_path])
    assert reads == [(5,17)]
    assert [f['meta']['tick'] for f in seq] == [10,11]
    assert seq[0]['person_keys'] == seq[1]['person_keys'] == [70,20]
    assert np.array_equal(seq[0]['img'][:6], seq[1]['img'][:6])
    assert (seq[0]['img'][-1] == 255).all()


class Head(nn.Module):
    def forward(self, features):
        return features[0]


class FakeYolo(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(17, 8, 1)
        self.model = nn.ModuleList([Head()])

    def forward(self, img):
        return self.model[-1]([self.conv(img)])


@pytest.fixture
def net(monkeypatch):
    monkeypatch.setattr(Q.S, 'build', lambda *a: FakeYolo())
    monkeypatch.setattr(Q.S, 'HP', 8)
    monkeypatch.setattr(Q.S, 'W', 8)
    return Q.DoorSegTemporal('cpu', hidden=4, embedding=6)


def test_adapter_old_checkpoint_new_roundtrip_and_context_guard(net):
    original = Q.S.DoorSeg.state(net)
    net.load(original)
    x = torch.randn(1, 17, 8, 8)
    with pytest.raises(RuntimeError, match='ordered'):
        net.yolo(x)
    net.prepare_frame(['a'], [0])
    result = net.yolo(x)
    assert torch.equal(result, net.yolo.conv(x))
    saved = net.state()
    net.load(saved)
    assert net.memory is None
    with torch.no_grad():
        net.temporal.residual.weight.fill_(3)
    net.load(original)
    assert torch.count_nonzero(net.temporal.residual.weight) == 0
    saved['temporal_arch'] = dict(saved['temporal_arch'], embedding=999)
    with pytest.raises(ValueError, match='architecture'):
        net.load(saved)


def test_sequence_loss_backpropagates_all_heads_and_releases_memory(net, monkeypatch):
    # Miniature collate, preserving original API. Real target geometry is tested
    # separately by the existing segmentation tests/Colab smoke.
    def collate(frames, dev):
        b = len(frames)
        mask = torch.zeros(b, 16, 16, dtype=torch.long)
        mask[:, :8] = 1
        mask[:, 8:] = 2
        batch = {'img': torch.ones(b, 17, 8, 8, dtype=torch.uint8), 'masks': mask,
                 'batch_idx': torch.arange(b).repeat_interleave(2)}
        return batch, torch.ones(b*2, 4), torch.ones(b*2, 2, dtype=torch.bool)
    monkeypatch.setattr(Q.S, 'collate', collate)
    seq = [{'namespace': ('clip','cam1'), 'meta': {'tick': k}, 'person_keys': [12,34]} for k in [0,1,2]]
    loss, stats = Q.sequence_loss(net, [seq], lambda pred,batch: (pred.square().mean(), None), 'cpu')
    loss.backward()
    assert torch.isfinite(loss) and stats == {'identity_anchors': 6, 'frames': 3}
    for module in [net.yolo, net.off, net.temporal.residual, net.temporal.fingerprint]:
        assert any(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0
                   for p in module.parameters())
    assert net.memory is None and net.embedding_map is None
    seq[1]['meta']['tick'] = 99
    with pytest.raises(ValueError, match='unordered'):
        Q.sequence_loss(net, [seq], None, 'cpu')
