"""Failure isolation for optimizer updates, including mutated BN and Adam."""
import copy
from pathlib import Path
import sys

import pytest
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'multicam'))
from door_seg_sequence_update import update


class Mini:
    def __init__(self):
        self.model = nn.Sequential(nn.Linear(2, 3), nn.BatchNorm1d(3), nn.Linear(3, 1))
        self.memory = None

    def parameters(self):
        return list(self.model.parameters())

    def state(self):
        return {'model': self.model.state_dict()}

    def load(self, state):
        self.model.load_state_dict(state['model'])

    def reset_memory(self):
        self.memory = None


def equal(a, b):
    if torch.is_tensor(a):
        return torch.equal(a, b)
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(equal(a[k], b[k]) for k in a)
    if isinstance(a, (tuple, list)):
        return len(a) == len(b) and all(equal(x, y) for x, y in zip(a, b))
    return a == b


def setup():
    torch.manual_seed(11)
    net = Mini(); opt = torch.optim.AdamW(net.parameters(), lr=.01)
    x = torch.tensor([[1., 2.], [3., -1.], [-2., .5]])
    def loss():
        net.memory = torch.ones(2)
        return net.model(x).square().mean(), {'frames': 3}
    update(net, opt, loss)
    return net, opt, loss


@pytest.mark.parametrize('fault', ['loss', 'gradient', 'optimizer', 'yield'])
def test_rejected_update_restores_weights_bn_and_existing_adam(fault):
    net, opt, loss = setup()
    before = copy.deepcopy(net.state()), copy.deepcopy(opt.state_dict())
    hooks = []
    if fault == 'gradient':
        hooks.append(net.parameters()[0].register_hook(lambda grad: grad * float('inf')))
    if fault == 'optimizer':
        original = opt.step
        def broken():
            original()
            with torch.no_grad(): net.parameters()[0].flatten()[0] = float('inf')
        opt.step = broken
    calls = []
    def guard():
        calls.append(1)
        if fault == 'yield' and len(calls) == 4:
            raise RuntimeError('yield after optimizer')
    def candidate():
        value, stats = loss()
        return value * float('inf') if fault == 'loss' else value, stats
    with pytest.raises((ValueError, RuntimeError)):
        update(net, opt, candidate, guard=guard)
    assert equal(net.state(), before[0]) and equal(opt.state_dict(), before[1])
    assert net.memory is None and all(p.grad is None for p in net.parameters())
    for hook in hooks: hook.remove()


def test_success_matches_control_and_does_not_reset_adam():
    net, opt, loss = setup()
    control = Mini(); control.load(copy.deepcopy(net.state()))
    other = torch.optim.AdamW(control.parameters(), lr=.01)
    other.load_state_dict(copy.deepcopy(opt.state_dict()))
    x = torch.tensor([[1., 2.], [3., -1.], [-2., .5]])
    other.zero_grad(); value = control.model(x).square().mean(); value.backward()
    torch.nn.utils.clip_grad_norm_(control.parameters(), 10, error_if_nonfinite=True); other.step()
    result = update(net, opt, loss)
    assert equal(net.state(), control.state()) and equal(opt.state_dict(), other.state_dict())
    assert result['frames'] == 3 and net.memory is None
