"""Experimental memory/fingerprint heads. Not enabled in doorseg_c_background.

State belongs to one ordered batch of camera streams. Callers own it; it is
never shared implicitly between clips or persisted in a model checkpoint.
"""
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class MemoryState:
    hidden: torch.Tensor
    streams: tuple
    ticks: tuple

    def detached(self):
        return MemoryState(self.hidden.detach(), self.streams, self.ticks)


def separable(cin, cout):
    return nn.Sequential(nn.Conv2d(cin, cin, 3, padding=1, groups=cin),
                         nn.Conv2d(cin, cout, 1))


class TemporalHeads(nn.Module):
    """Stride-16 ConvGRU, zero-initialized P3 residual, stride-8 fingerprint.

    A changed stream, missing tick or incompatible tensor resets that lane.
    retain_graph=True is for a bounded training sequence; inference detaches.
    """
    def __init__(self, channels, hidden=48, embedding=64):
        super().__init__()
        self.arch = dict(version=1, channels=channels, hidden=hidden, embedding=embedding)
        self.project = nn.Conv2d(channels, hidden, 1)
        self.gates = separable(hidden * 2, hidden * 2)
        self.candidate = separable(hidden * 2, hidden)
        self.residual = nn.Conv2d(hidden, channels, 1)
        nn.init.zeros_(self.residual.weight)
        nn.init.zeros_(self.residual.bias)
        self.fingerprint = nn.Sequential(nn.Conv2d(channels, hidden, 1), nn.SiLU(),
                                         separable(hidden, embedding))

    def forward(self, features, streams, ticks, state=None, *, retain_graph=False):
        streams, ticks = tuple(streams), tuple(int(t) for t in ticks)
        if len(streams) != features.shape[0] or len(ticks) != len(streams):
            raise ValueError('one stream and tick per batch lane required')
        # Pool before the recurrent convolutions; no full-resolution memory.
        x = F.avg_pool2d(self.project(features), 2, ceil_mode=True)
        h = torch.zeros_like(x)
        if state is not None and (state.hidden.shape == x.shape and
                                  state.hidden.device == x.device and state.hidden.dtype == x.dtype):
            keep = [a == b and t == p + 1 for a, b, t, p in
                    zip(streams, state.streams, ticks, state.ticks)]
            if len(keep) == len(streams):
                previous = state.hidden if retain_graph else state.hidden.detach()
                mask = torch.tensor(keep, device=x.device, dtype=torch.bool).view(-1, 1, 1, 1)
                h = torch.where(mask, previous, torch.zeros_like(previous))
        reset, update = self.gates(torch.cat((x, h), 1)).sigmoid().chunk(2, 1)
        proposal = self.candidate(torch.cat((x, reset * h), 1)).tanh()
        hidden = (1 - update) * h + update * proposal
        enhanced = features + F.interpolate(self.residual(hidden), features.shape[-2:], mode='nearest')
        dense = self.fingerprint(enhanced)
        new_state = MemoryState(hidden, streams, ticks)
        if not retain_graph:
            new_state = new_state.detached()
        return enhanced, dense, new_state


def pool_instances(dense, masks, max_instances=254):
    """Pool local instance IDs in a BxHxW map, excluding background/empty IDs.

    Return unit embeddings, batch lanes, local IDs; the same local ID in two
    frames is not an identity label. Work/temporary storage scales with pixels,
    rather than persons times pixels. Overlap follows the segmenter's ID map.
    """
    b, channels, h, w = dense.shape
    idx = F.interpolate(masks[:, None].float(), (h, w), mode='nearest')[:, 0].long()
    if masks.shape[0] != b or torch.any(idx < 0) or torch.any(idx > max_instances):
        raise ValueError('invalid local instance map')
    stride = max_instances + 1
    groups = idx + torch.arange(b, device=dense.device)[:, None, None] * stride
    values = dense.float().permute(0, 2, 3, 1).reshape(-1, channels)
    sums = values.new_zeros(b * stride, channels).index_add(0, groups.flatten(), values)
    counts = values.new_zeros(b * stride).index_add(0, groups.flatten(), values.new_ones(b * h * w))
    ids = torch.arange(b * stride, device=dense.device)
    visible = (counts > 0) & (ids % stride != 0)
    ids = ids[visible]
    embeddings = F.normalize(sums[visible] / counts[visible, None], dim=1, eps=1e-6)
    return embeddings, ids // stride, ids % stride


def pool_prediction_masks(dense, masks):
    """One frame's independent predicted masks, preserving overlap between people.

    Result includes a validity bit: an empty/zero-vector crop must not be used
    for identity association. Original detector output tuples remain unchanged.
    """
    if dense.shape[0] != 1 or masks.ndim != 3:
        raise ValueError('expected one feature frame and NxHxW masks')
    if not len(masks):
        return dense.new_empty((0, dense.shape[1])), torch.empty(0, dtype=torch.bool, device=dense.device)
    small = F.interpolate(masks[:, None].float(), dense.shape[-2:], mode='nearest')[:, 0]
    counts = small.sum((1, 2))
    vectors = small.flatten(1) @ dense[0].float().flatten(1).T / counts[:, None].clamp_min(1)
    valid = (counts > 0) & (vectors.norm(dim=1) > 1e-6)
    return F.normalize(vectors, dim=1, eps=1e-6), valid


def identity_loss(embeddings, identities, namespaces, ticks, temperature=0.1):
    """Positives across ticks, negatives only different co-visible identities.

    ID namespaces include recording/camera and teacher session unless IDs have
    already been glued by the approved spatial/owner procedure. Unlinked IDs
    across time or clips are not assumed to be different people. Unknown=-1.
    """
    if temperature <= 0:
        raise ValueError('temperature must be positive')
    n = embeddings.shape[0]
    if any(len(v) != n for v in (identities, namespaces, ticks)):
        raise ValueError('identity metadata length mismatch')
    if n < 2:
        return embeddings.sum() * 0, 0
    device = embeddings.device
    ids = torch.as_tensor(identities, device=device)
    ns = torch.as_tensor(namespaces, device=device)
    ts = torch.as_tensor(ticks, device=device)
    known = (ids[:, None] >= 0) & (ids[None, :] >= 0)
    same_ns = ns[:, None] == ns[None, :]
    positives = known & same_ns & (ids[:, None] == ids[None, :]) & (ts[:, None] != ts[None, :])
    negatives = known & same_ns & (ids[:, None] != ids[None, :]) & (ts[:, None] == ts[None, :])
    anchors = positives.any(1) & negatives.any(1)
    if not anchors.any():
        return embeddings.sum() * 0, 0
    logits = F.normalize(embeddings.float(), dim=1, eps=1e-6) @ F.normalize(embeddings.float(), dim=1, eps=1e-6).T
    logits = logits[anchors] / temperature
    pos, allowed = positives[anchors], (positives | negatives)[anchors]
    normalizer = logits.masked_fill(~allowed, -torch.inf).logsumexp(1)
    mean_positive = logits.masked_fill(~pos, 0).sum(1) / pos.sum(1)
    return (normalizer - mean_positive).mean(), int(anchors.sum())
