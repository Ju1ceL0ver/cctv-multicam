"""Ordered TRAIN sequences for the experimental memory/fingerprint heads.

No new trainer is started here; C's random single-frame trainer is unchanged.
"""
import random

import numpy as np
import torch

import door_seg as S
from door_seg_temporal import TemporalHeads, pool_instances, identity_loss


def sequence_starts(source, length=4):
    if length < 2:
        raise ValueError('a sequence needs at least two frames')
    available = set(source.starts)
    return [(session, tick) for session, tick in source.starts
            if all((session, tick + j) in available for j in range(length))]


def sample_sequence(source, length=4, seed=0, at=None, *, allowed_sources=None):
    """Use existing masks/offset labels and one video seek; reject TEST sources.

    RGB/background brightness uses one gain for the sequence. Spoiled previous
    masks remain the existing teacher augmentation. No synthetic ID gluing.
    """
    allowed = S.sources() if allowed_sources is None else allowed_sources
    if source.d.resolve() not in {p.resolve() for p in allowed}:
        raise ValueError('sequence source is not in the TRAIN manifest')
    starts = sequence_starts(source, length)
    rng = random.Random(seed)
    session, first = at if at is not None else rng.choice(starts)
    if (session, first) not in starts:
        raise ValueError('sequence crosses a gap or session boundary')
    cache = source.read(first - max(S.PAST), first + length + max(S.FUT))
    if any(tick not in cache for tick in range(first - max(S.PAST), first + length + max(S.FUT))):
        raise ValueError('incomplete video context')
    gain = rng.uniform(0.8, 1.2) if rng.random() < 0.5 else 1.0
    frames = []
    for tick in range(first, first + length):
        frame = S.sample(source, rng, train=False, at=(session, tick), cache=cache)
        if frame is None:
            raise ValueError('incomplete video context')
        before = source.people(session, tick - 1)
        frame['img'][-1, :S.H] = S.spoil([m for m, _ in before.values()], rng) * 255
        frame['img'][:6] = np.clip(frame['img'][:6].astype(np.float32) * gain, 0, 255).astype(np.uint8)
        people = source.people(session, tick)
        order = sorted(people, key=lambda key: -int(people[key][0].sum()))[:254]
        # Preserve original keys; batch-level encoding must not confuse two
        # different local sequence mappings from the same recording.
        frame['person_keys'] = order
        frames.append(frame)
    namespace = (str(source.d.resolve()), session)
    for frame in frames:
        frame['namespace'] = namespace
    return frames


class DoorSegTemporal(S.DoorSeg):
    """Separate experimental adapter; legacy prediction API is preserved.

    Every YOLO call requires prepare_frame. This prevents the single-frame eval
    or old shuffled trainer from silently evaluating recurrent state wrongly.
    """
    def __init__(self, dev, weights='yolo26s-seg.pt', hidden=48, embedding=64):
        super().__init__(dev, weights)
        self.temporal = TemporalHeads(self.feat.shape[1], hidden, embedding).to(dev)
        self.memory = None
        self.embedding_map = None
        self.context = None
        self.yolo.model[-1].register_forward_pre_hook(self._enhance)

    def prepare_frame(self, streams, ticks, *, retain_graph=False):
        if self.context is not None:
            raise RuntimeError('previous frame context was not consumed')
        self.context = (tuple(streams), tuple(ticks), retain_graph)

    def reset_memory(self):
        self.memory, self.context, self.embedding_map = None, None, None

    def _enhance(self, module, args):
        if self.context is None:
            raise RuntimeError('temporal YOLO requires ordered prepare_frame context')
        streams, ticks, retain_graph = self.context
        self.context = None
        features = list(args[0])
        features[0], self.embedding_map, self.memory = self.temporal(
            features[0], streams, ticks, self.memory, retain_graph=retain_graph)
        self.feat = features[0]
        return (features,) + args[1:]

    def parameters(self):
        return super().parameters() + list(self.temporal.parameters())

    def state(self):
        return dict(super().state(), temporal=self.temporal.state_dict(), temporal_arch=self.temporal.arch)

    def load(self, state):
        if 'temporal' in state and state.get('temporal_arch') != self.temporal.arch:
            raise ValueError('temporal checkpoint architecture differs')
        super().load(state)
        if 'temporal' in state:
            self.temporal.load_state_dict(state['temporal'])
        else:
            arch = self.temporal.arch
            fresh = TemporalHeads(arch['channels'], arch['hidden'], arch['embedding'])
            self.temporal.load_state_dict(fresh.state_dict())
        self.reset_memory()


def _float_predictions(value):
    if torch.is_tensor(value):
        return value.float() if value.is_floating_point() else value
    if isinstance(value, dict):
        return {k: _float_predictions(v) for k, v in value.items()}
    if isinstance(value, tuple):
        return tuple(_float_predictions(v) for v in value)
    if isinstance(value, list):
        return [_float_predictions(v) for v in value]
    return value


def sequence_loss(net, sequences, criterion, dev, identity_weight=0.1, *, before_frame=None):
    """Batched ordered unroll: original segmentation/offset + identity loss.

    Returns one differentiable scalar; caller owns autocast, finite checks,
    clipping, optimizer, evaluation and persistence. Detach state after loss to
    bound graphs between sequences, without truncating this sequence's BPTT.
    """
    if not sequences or len({len(seq) for seq in sequences}) != 1 or len(sequences[0]) < 2:
        raise ValueError('equal non-empty sequences of at least two frames required')
    if identity_weight < 0:
        raise ValueError('identity weight cannot be negative')
    for seq in sequences:
        stream = seq[0]['namespace']
        first = seq[0]['meta']['tick']
        if any(frame['namespace'] != stream or frame['meta']['tick'] != first + j
               for j, frame in enumerate(seq)):
            raise ValueError('unordered sequence or changed stream')
    net.reset_memory()
    embeddings, ids, namespaces, ticks, losses = [], [], [], [], []
    streams = [seq[0]['namespace'] for seq in sequences]
    namespace_ids = {key: i for i, key in enumerate(dict.fromkeys(streams))}
    person_ids = {}
    for seq in sequences:
        for frame in seq:
            for key in frame['person_keys']:
                person_ids.setdefault((frame['namespace'], key), len(person_ids))
    try:
        for j in range(len(sequences[0])):
            if before_frame is not None:
                before_frame()
            frames = [seq[j] for seq in sequences]
            batch, off, offv = S.collate(frames, dev)
            net.prepare_frame(streams, [frame['meta']['tick'] for frame in frames], retain_graph=True)
            pred = net.yolo(batch['img'].float() / 255)
            offsets = net.off(net.feat)
            # AMP convolutions, explicit FP32 loss arithmetic and mask products.
            with torch.autocast(torch.device(dev).type, enabled=False):
                seg, _ = criterion(_float_predictions(pred), batch)
                losses.append(seg.float().sum() + 2 * len(frames) * S.offset_loss(offsets.float(), batch, off, offv))
            vector, lanes, local_ids = pool_instances(net.embedding_map, batch['masks'])
            embeddings.append(vector)
            # Only the small metadata lookup crosses to CPU; not feature pixels.
            for lane, local_id in zip(lanes.tolist(), local_ids.tolist()):
                frame = frames[lane]
                ids.append(person_ids[(frame['namespace'], frame['person_keys'][local_id - 1])])
                namespaces.append(namespace_ids[streams[lane]])
                ticks.append(frame['meta']['tick'])
        with torch.autocast(torch.device(dev).type, enabled=False):
            reid, anchors = identity_loss(torch.cat(embeddings).float(), ids, namespaces, ticks)
            base = torch.stack(losses).float().mean()
            return base + identity_weight * reid, {'identity_anchors': anchors, 'frames': len(losses) * len(sequences), 'loss_dtype': str(base.dtype), 'seg_offset_loss': float(base.detach()), 'identity_loss': float(reid.detach())}
    finally:
        net.reset_memory()
