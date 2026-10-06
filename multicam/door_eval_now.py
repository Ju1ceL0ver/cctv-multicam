"""06.10: the rule fitted on the teacher's tracks of 17 and 18.09 (live features only), applied to 19.09 -- the
teacher's tracks and the small SAM's -- against the owner's truth with his /doormark marks."""
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PY = sys.executable
env = dict(os.environ, CUDA_VISIBLE_DEVICES='-1', PYTHONIOENCODING='utf-8', RA_DOOR_SKIP='20260918 15:06-15:29',
           RA_DOOR_NOCNN='1', RA_DOOR_NODEPTH='1', RA_DOOR_NOLK='1', RA_DOOR_NOTAP='1')
run = lambda args: print(subprocess.run([PY] + args, cwd=str(ROOT), env=env, capture_output=True, text=True).stdout[-2500:], flush=True)
run(['door_rule.py', 'fit', 'sam31_live_1718', '20260917=data/door_v2/20260917_sam31.jsonl.gz', '20260918=data/door_v2/20260918_sam31.jsonl.gz'])
run(['door_eval_rule.py', 'sam31_live_1718', '20260919=data/door_v2/20260919_sam31.jsonl.gz'])
micro = ROOT / 'data' / 'door_v2' / '20260919_micro1s3.jsonl.gz'
while not micro.exists():
    time.sleep(30)
time.sleep(10)
run(['door_eval_rule.py', 'sam31_live_1718', '20260919=data/door_v2/20260919_micro1s3.jsonl.gz'])
