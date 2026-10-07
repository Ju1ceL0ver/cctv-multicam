"""07.10.2026: camera 1's /inout answers as one file for training elsewhere -> data/inout/io_cam1.npz (CPU).
X crop 160x96x7, F frame 180x320x7 (RGB, person mask, shop floor, x, y), G 10 geometry numbers, y class, day, door, ids."""
import numpy as np
import io_fast as I
X, F, G, y, day, door, ids = I.dataset()
np.savez_compressed(I.ROOT / 'data' / 'inout' / 'io_cam1.npz', X=X, F=F, G=G, y=y, day=day, door=door, ids=np.array(ids))
print('saved', len(y), flush=True)
