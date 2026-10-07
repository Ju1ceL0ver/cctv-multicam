"""Prototype of door-gated shop membership, separate from frame classifier.

Caller supplies a TRUSTED side observation (0 outside, 1 inside, None when
occluded/uncertain) and contact with the physical doorway. Door contact must
not be inferred from the free-floor distance or simply from class 'doorway'.
Tracking IDs must refer to the same person. No measured accuracy claim.
"""
from dataclasses import dataclass

@dataclass
class State:
    side: int | None = None
    pending: int | None = None
    confirmations: int = 0
    armed_at: float | None = None
    seen_at: float = 0.0

class ShopSideState:
    def __init__(self, confirmations=3, crossing_window=30.0, max_gap=60.0):
        self.required=confirmations
        self.crossing_window=crossing_window
        self.max_gap=max_gap
        self.tracks={}

    def update(self, track_id, t, trusted_side=None, door_contact=False):
        """Return (stable_side, entry_or_exit_or_None). Initial sighting is not an event."""
        if trusted_side not in (None,0,1):
            raise ValueError('trusted_side must be None, 0 or 1')
        s=self.tracks.get(track_id)
        if s is not None and t<s.seen_at:
            raise ValueError('observations must be chronological')
        if s is None or t-s.seen_at>self.max_gap:
            s=self.tracks[track_id]=State(seen_at=t)
        s.seen_at=t
        if door_contact:
            s.armed_at=t
        if trusted_side is None:
            return s.side,None
        if trusted_side==s.side:
            s.pending=None;s.confirmations=0
            return s.side,None
        armed=s.armed_at is not None and t-s.armed_at<=self.crossing_window
        if s.side is not None and not armed:
            s.pending=None;s.confirmations=0
            return s.side,None
        if s.pending!=trusted_side:
            s.pending=trusted_side;s.confirmations=0
        s.confirmations+=1
        if s.confirmations<self.required:
            return s.side,None
        previous=s.side;s.side=trusted_side;s.pending=None;s.confirmations=0;s.armed_at=None
        return s.side,None if previous is None else ('entry' if s.side==1 else 'exit')
