"""Run a list of detection jobs one after another (one GPU)."""
import os, sys, subprocess, time
ROOT = os.path.dirname(os.path.abspath(__file__))
PY = r'C:\Users\ArykovAA\AppData\Local\miniconda3\envs\cctv_base\python.exe'
JOBS = [
    ('c155236', '20260917', '15:52:36', 60),
    ('c103700', '20260917', '10:37:00', 570),
    ('c171400', '20260917', '17:14:00', 540),
    ('c183400', '20260917', '18:34:00', 480),
]
log = open(os.path.join(ROOT, 'data', 'queue_detect.log'), 'a')
for clip, day, hms, sec in JOBS:
    log.write('%s start %s\n' % (time.strftime('%H:%M:%S'), clip)); log.flush()
    r = subprocess.run([PY, os.path.join(ROOT, 'detect_raw.py'), clip, day, hms, str(sec)], capture_output=True, text=True)
    log.write('%s end %s rc=%s %s\n' % (time.strftime('%H:%M:%S'), clip, r.returncode, r.stderr[-600:])); log.flush()
