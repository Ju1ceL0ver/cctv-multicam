"""Experimental removal of bit-identical nonempty predicted masks.

No overlap threshold, box merge or ID merge. Retain one original prediction
with the highest confidence; ties retain the first. Distinct masks, including
nearly identical masks, remain separate. The foreground union is unchanged.
REJECTED on dense TRAIN200: fewer breaks, contaminated observations17->18.
Retained only for reproducing that experiment. Not enabled in inference,
production or C training.
"""
import numpy as np


def exact_masks(predictions):
    winners = {}
    independent = []
    for i, p in enumerate(predictions):
        mask = np.asarray(p[0])
        if mask.dtype != np.bool_ or mask.ndim != 2:
            raise ValueError('expected two-dimensional boolean prediction mask')
        if not mask.any():
            independent.append(i)
            continue
        key = (mask.shape, mask.tobytes())
        previous = winners.get(key)
        if previous is None or p[2] > predictions[previous][2]:
            winners[key] = i
    retained = sorted(independent + list(winners.values()))
    return [predictions[i] for i in retained], retained
