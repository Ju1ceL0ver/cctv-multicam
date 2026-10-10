"""One transactional update for a separate bounded sequence experiment.

No trainer starts here. A rejected loss, gradient, optimizer result or yield
restores parameters, buffers and Adam state; the successful-step counter is
caller-owned. Ephemeral memory is cleared on every path.
"""
import copy

import torch

from door_seg_sequence_state import finite


def update(net, optimizer, loss_fn, *, guard=lambda: None, max_norm=10.):
    if max_norm <= 0 or not torch.isfinite(torch.tensor(max_norm)):
        raise ValueError('finite positive gradient limit required')
    guard()
    finite(net.state()); finite(optimizer.state_dict())
    before_model = copy.deepcopy(net.state())
    before_optimizer = copy.deepcopy(optimizer.state_dict())
    try:
        optimizer.zero_grad(set_to_none=True)
        loss, stats = loss_fn()
        finite(loss, 'sequence_loss')
        guard()
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm, error_if_nonfinite=True)
        guard()
        optimizer.step()
        finite(net.state()); finite(optimizer.state_dict())
        guard()
        return {'loss': float(loss.detach()), 'grad_norm': float(norm), **stats}
    except BaseException:
        net.load(before_model)
        optimizer.load_state_dict(before_optimizer)
        raise
    finally:
        optimizer.zero_grad(set_to_none=True)
        net.reset_memory()
