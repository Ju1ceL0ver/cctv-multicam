"""Smoke check: the click segmenter of /crowdmasks on a real door frame (processor)."""
import sys, time, os
import cv2, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import day_masks
cap = cv2.VideoCapture(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'sam31_door', 'door_20260918_18720', 'cam1', 'video.mp4'))
cap.set(cv2.CAP_PROP_POS_FRAMES, int(63 * 12.5)); ok, f = cap.read()
f = cv2.resize(f, (1280, 720))
s = day_masks.Segmenter(); print('model', s.name)
t0 = time.time(); s.encode('x', 1, f); t1 = time.time()
m = s.mask('x', [[300, 330]], [1]); t2 = time.time()
print('encode %.1f s, click %.2f s, mask px %d' % (t1 - t0, t2 - t1, 0 if m is None else int(m.sum())))
