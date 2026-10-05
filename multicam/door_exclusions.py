"""Camera-specific non-customer regions, in normalized image coordinates.

The neighbouring storefront is excluded by the bottom of the mask, so people
walking in the mall corridor remain available for detecting actual entries.
The advertising panel is excluded by mask occupancy, not a person's identity.
"""
import cv2
import numpy as np

# Measured on cam1's 1280 x 720 image, 05.10.2026.
NEIGHBOUR = np.array([(0, 0), (470, 0), (470, 140), (380, 155), (0, 310)], np.float32) / [1280, 720]
POSTER = (514 / 1280, 35 / 720, 559 / 1280, 158 / 720)


def excluded(mask, cam='cam1'):
    if cam != 'cam1':
        return False
    ys, xs = np.nonzero(mask)
    if not len(ys):
        return False
    h, w = mask.shape
    bottom = ys >= ys.max() - 2
    foot = (float(np.median(xs[bottom])) / w, float(ys.max()) / h)
    if cv2.pointPolygonTest(NEIGHBOUR.astype(np.float32), foot, False) >= 0:
        return True
    x1, y1, x2, y2 = POSTER
    on_panel = (xs / w >= x1) & (xs / w <= x2) & (ys / h >= y1) & (ys / h <= y2)
    return bool(on_panel.mean() >= 0.65)
