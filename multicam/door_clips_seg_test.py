"""Smoke test of door_clips_seg.Tape on synthetic frames (no camera, no GPU)."""
import sys, time, os
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import door_clips_seg as D
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'visits_lab', 'cliptest')
tp = D.Tape(out); tp.start()
t0 = time.time() - 100
for i in range(300):
    fr = np.full((720, 1280, 3), (i % 255, 80, 120), np.uint8)
    m = np.zeros((180, 320), bool); m[40:150, 20 + i:60 + i] = 1
    tp.put(t0 + i * 0.08, fr, [(7, m), (9, np.roll(m, 100, 1))])
    if i == 100:
        tp.ask({'t': t0 + i * 0.08, 'w': '7', 'kind': 'out', 'clock': '2026-10-14 16:50:00'}, 'test_out')
    time.sleep(0.01)
time.sleep(20)
p = os.path.join(out, time.strftime('%Y%m%d', time.localtime(t0)), 'test_out.mp4')
print(os.path.exists(p), os.path.exists(p) and os.path.getsize(p))
c = os.path.join(os.path.dirname(out), 'crowd', time.strftime('%Y%m%d', time.localtime(t0)), 'test_out')
print('crowd', os.path.isdir(c) and len(os.listdir(c)))
