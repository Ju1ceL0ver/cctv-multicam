"""Door side inference for a coarse student mask; production settings stay intact.

Nearest-neighbour expansion of a stride-4 mask adds up to a cell's extent to
its deepest pixel. Require that extra margin for an unequivocal inside vote.
The CPU RGB classifier still votes on observations in the uncertain band.
"""
from copy import deepcopy
import math
from door_combo import Live as BaseLive


def quantization_margin(line, stride=4.0):
    if line is None:
        return 0.0
    dx=line['p2'][0]-line['p1'][0]
    dy=line['p2'][1]-line['p1'][1]
    length=math.hypot(dx,dy)
    if not length:
        raise ValueError('door line has zero length')
    return float(stride)*(abs(dx)+abs(dy))/length


class Live(BaseLive):
    def __init__(self):
        super().__init__()
        self.quantization_margin=quantization_margin(self.line)
        if self.side is not None:
            self.side=deepcopy(self.side)
            self.side['h_in']+=self.quantization_margin
