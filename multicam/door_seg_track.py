"""Online association for door_seg.predict outputs; no training or GPU dependencies.

Offsets are input pixels per ONE video tick (0.08 s), not mask pixels.
Each update receives an actual tick index. Missing observations never produce masks.
Appearance is reserved for a later trained head; this baseline uses both offset
directions, shifted masks and short, unambiguous motion continuations only.
"""
from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment


@dataclass(frozen=True)
class Config:
    max_gap: int = 25
    max_step: float = 64.0
    centre_gate: float = 0.3
    gap_growth: float = 0.07
    max_cost: float = 0.85
    ambiguity: float = 0.10


def centre(box):
    return (np.asarray(box[:2]) + np.asarray(box[2:])) / 2


def shifted_iou(a, b, delta):
    """Masks at stride 4; integer translation without wrapping around the image."""
    dx, dy = np.rint(np.asarray(delta) / 4).astype(int)
    h, w = a.shape
    if abs(dx) >= w or abs(dy) >= h:
        return 0.0
    ax, ay = max(0, -dx), max(0, -dy)
    bx, by = max(0, dx), max(0, dy)
    ww, hh = w - abs(dx), h - abs(dy)
    inter = np.logical_and(a[ay:ay + hh, ax:ax + ww], b[by:by + hh, bx:bx + ww]).sum()
    return float(inter / max(1, int(a.sum() + b.sum() - inter)))


@dataclass
class Track:
    pid: int
    tick: int
    pred: tuple
    velocity: np.ndarray


class Tracker:
    def __init__(self, config=None):
        self.config = config or Config()
        self.tracks = {}
        self.next_id = 1
        self.tick = None
        self.stats = dict(births=0, adjacent=0, recovered=0, ambiguous=0)

    def cost(self, track, pred, tick):
        c = self.config
        gap = tick - track.tick
        old, new = centre(track.pred[1]), centre(pred[1])
        height = max(16.0, min(track.pred[1][3] - track.pred[1][1], pred[1][3] - pred[1][1]))
        if np.linalg.norm(new - old) > c.max_step * gap:
            return np.inf
        if gap == 1:
            # Both heads must point towards the same person. A single good
            # direction cannot override the other direction pointing elsewhere.
            forward = old + np.asarray(track.pred[4])
            backward = new + np.asarray(pred[3])
            err = max(np.linalg.norm(new - forward), np.linalg.norm(old - backward))
            gate = max(8.0, c.centre_gate * height)
            if err > gate:
                return np.inf
            overlap = shifted_iou(track.pred[0], pred[0], track.pred[4])
            return 0.75 * err / gate + 0.25 * (1 - overlap)
        # One-step heads are NOT multiplied over an unseen interval. Use bounded
        # observed velocity, and allow recovery only when both sides are unique.
        expected = old + track.velocity * gap
        gate = min(c.max_step * gap, (c.centre_gate + c.gap_growth * gap) * height)
        err = np.linalg.norm(new - expected)
        if err > gate:
            return np.inf
        return 0.75 * err / gate + 0.10 + 0.15 * gap / c.max_gap

    def update(self, tick, predictions):
        """Return [(pid, prediction)] in input order; strictly increasing ticks."""
        if self.tick is not None and tick <= self.tick:
            raise ValueError('ticks must increase; create a new Tracker for a new stretch')
        self.tick = tick
        c = self.config
        self.tracks = {pid: t for pid, t in self.tracks.items() if tick - t.tick <= c.max_gap}
        old = list(self.tracks.values())
        predictions = list(predictions)
        costs = np.full((len(old), len(predictions)), np.inf)
        for i, tr in enumerate(old):
            for j, p in enumerate(predictions):
                costs[i, j] = self.cost(tr, p, tick)
        allowed = costs <= c.max_cost
        # Refuse uncertain identity choices BEFORE global assignment, so the
        # Hungarian solver cannot turn a rejected choice into a forced second.
        for i, j in zip(*np.nonzero(allowed)):
            others_row = np.delete(costs[i], j)
            others_col = np.delete(costs[:, j], i)
            rival = min(np.min(others_row, initial=np.inf), np.min(others_col, initial=np.inf))
            if rival <= c.max_cost and rival - costs[i, j] < c.ambiguity:
                allowed[i, j] = False
                self.stats['ambiguous'] += 1
        linked = {}
        if old and predictions:
            # One private dummy per track: choosing a new ID is always possible.
            matrix = np.full((len(old), len(predictions) + len(old)), 1e6)
            matrix[:, :len(predictions)] = np.where(allowed, costs, 1e6)
            for i in range(len(old)):
                matrix[i, len(predictions) + i] = c.max_cost + 1e-6
            ii, jj = linear_sum_assignment(matrix)
            linked = {j: old[i] for i, j in zip(ii, jj) if j < len(predictions) and allowed[i, j]}
        out = []
        for j, p in enumerate(predictions):
            if j in linked:
                tr = linked[j]
                gap = tick - tr.tick
                v = (centre(p[1]) - centre(tr.pred[1])) / gap
                speed = np.linalg.norm(v)
                if speed > c.max_step:
                    v = v * c.max_step / speed
                pid = tr.pid
                self.stats['adjacent' if gap == 1 else 'recovered'] += 1
            else:
                pid, v = self.next_id, np.zeros(2)
                self.next_id += 1
                self.stats['births'] += 1
            self.tracks[pid] = Track(pid, tick, p, v)
            out.append((pid, p))
        return out
