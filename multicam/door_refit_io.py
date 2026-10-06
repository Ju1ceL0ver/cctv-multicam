"""06.10, after the owner's door batch on /inout: refit the inside/outside classifier on every answer, then the door rule
on the teacher's and the small SAM's tracks (stitched), day against day, against the owner's truth with his marks."""
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PY = sys.executable
env = dict(os.environ, CUDA_VISIBLE_DEVICES='-1', PYTHONIOENCODING='utf-8', RA_DOOR_SKIP='20260918 15:06-15:29',
           RA_DOOR_NOCNN='1', RA_DOOR_NODEPTH='1', RA_DOOR_NOLK='1', RA_DOOR_NOTAP='1', RA_DOOR_STITCH='1')
run = lambda args: print(subprocess.run([PY] + args, cwd=str(ROOT), env=env, capture_output=True, text=True).stdout[-2500:], flush=True)
print('== classifier', flush=True)
run(['io_door_eval.py', '--refit'])
print('== teacher tracks 17/18/19', flush=True)
run(['door_rule.py', 'fit', 'sam31_live2', '20260917=data/door_v2/20260917_sam31.jsonl.gz', '20260918=data/door_v2/20260918_sam31.jsonl.gz',
     '20260919=data/door_v2/20260919_sam31.jsonl.gz'])
print('== teacher tracks 17/19', flush=True)
run(['door_rule.py', 'fit', 'sam31_live2_1719', '20260917=data/door_v2/20260917_sam31.jsonl.gz', '20260919=data/door_v2/20260919_sam31.jsonl.gz'])
print('== small SAM tracks 17/19', flush=True)
run(['door_rule.py', 'fit', 'micro1s3_live2', '20260917=data/door_v2/20260917_micro1s3.jsonl.gz', '20260919=data/door_v2/20260919_micro1s3.jsonl.gz'])
