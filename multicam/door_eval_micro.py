"""06.10, the small SAM's own door rule: as soon as its runs of 17.09 and 19.09 exist, the rule fitted on one day of
its tracks and scored on the other (door_rule.fit's out-of-fold), and the debounce rule on both trackers."""
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PY = sys.executable
env = dict(os.environ, CUDA_VISIBLE_DEVICES='-1', PYTHONIOENCODING='utf-8', RA_DOOR_SKIP='20260918 15:06-15:29',
           RA_DOOR_NOCNN='1', RA_DOOR_NODEPTH='1', RA_DOOR_NOLK='1', RA_DOOR_NOTAP='1')
run = lambda args, e=env: print(subprocess.run([PY] + args, cwd=str(ROOT), env=e, capture_output=True, text=True).stdout[-3000:], flush=True)
p17 = ROOT / 'data' / 'door_v2' / '20260917_micro1s3.jsonl.gz'
while not p17.exists():
    time.sleep(30)
time.sleep(10)
print('== the small SAM rule, 17 <-> 19', flush=True)
run(['door_rule.py', 'fit', 'micro1s3_1719', '20260917=data/door_v2/20260917_micro1s3.jsonl.gz', '20260919=data/door_v2/20260919_micro1s3.jsonl.gz'])
print('== the teacher rule, 17 <-> 19 (same days, for comparison)', flush=True)
run(['door_rule.py', 'fit', 'sam31_live_1719', '20260917=data/door_v2/20260917_sam31.jsonl.gz', '20260919=data/door_v2/20260919_sam31.jsonl.gz'])
print('== debounce', flush=True)
run(['door_debounce.py'], dict(env, RA_DEB_RUNS='sam=sam31;micro=micro1s3', RA_DEB_DAYS='20260917,20260919', RA_DOOR_SKIP=''))
