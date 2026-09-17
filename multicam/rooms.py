"""Hand-traced shop-floor regions, one-time per camera (full-res pixel coords).

Traced on the median background on a 100 px grid. They exclude display stands,
the desk and the mall gallery seen through the glass (a different floor grid)."""
import numpy as np

_D = 1.6   # traced at 1600x900 display scale
FLOOR = {
    'cam1': [
        [(310, 505), (365, 490), (625, 352), (690, 340), (640, 430), (632, 600), (720, 615), (860, 740),
         (1000, 770), (1110, 795), (1245, 690), (1255, 600), (1250, 450), (1350, 430), (1365, 580),
         (1560, 600), (1560, 690), (1260, 735), (1175, 900), (660, 900)],
    ],
    'cam2': [
        [(630, 250), (750, 268), (760, 330), (850, 390), (880, 480), (870, 600), (850, 650), (870, 900),
         (650, 900), (560, 760), (455, 590), (620, 540), (625, 330)],
    ],
}


def polygons(cam):
    return [np.array(p, np.float32) * _D for p in FLOOR[cam]]


def mask(cam, shape=(1440, 2560)):
    import cv2
    m = np.zeros(shape, np.uint8)
    for p in polygons(cam):
        cv2.fillPoly(m, [p.astype(np.int32)], 255)
    return m
