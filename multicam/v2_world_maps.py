"""Where on the floor each point of a camera's picture is (PLANET's world-grounded positions): for a grid the size
of the ViT's stride-8 map (160 x 90 over the 2560 x 1440 frame), the floor point (x, y m, the common frame)
of the pixel's ray, and whether the ray meets the floor at all -> data/scene/world_<cam>.npy (90 x 160 x 3)."""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))


def main():
    import person3d
    calib = json.load(open(ROOT / 'data' / 'calib_final.json'))
    reg = json.load(open(ROOT / 'data' / 'cam1_to_cam2_affine.json'))
    for cam in ('cam1', 'cam2'):
        c = person3d.Camera(cam, calib)
        gy, gx = np.mgrid[0:90, 0:160]
        px = np.stack([(gx + 0.5) * 2560 / 160, (gy + 0.5) * 1440 / 90], -1).reshape(-1, 2)
        xy = c.on_plane(px) * person3d.UNIT
        if cam == 'cam1':
            xy = person3d.map_cam1(xy, reg)
        ok = np.isfinite(xy).all(1) & (np.abs(xy) < 40).all(1)          # far beyond the hall: the ray is near the horizon
        xy = np.where(ok[:, None], xy, 0.0)
        out = np.concatenate([xy, ok[:, None].astype(np.float64)], 1).reshape(90, 160, 3).astype(np.float32)
        np.save(ROOT / 'data' / 'scene' / ('world_%s.npy' % cam), out)
        print(cam, 'floor rays %.0f %%' % (100 * ok.mean()), 'x %.1f..%.1f y %.1f..%.1f' % (xy[ok, 0].min(), xy[ok, 0].max(), xy[ok, 1].min(), xy[ok, 1].max()))


if __name__ == '__main__':
    main()
