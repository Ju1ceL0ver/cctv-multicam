"""06.10: does stitching broken tracks help the door rule? The small SAM's and the teacher's rule, 17 <-> 19, with and
without door_stitch, against the owner's truth with his /doormark marks."""
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PY = sys.executable
base = dict(os.environ, CUDA_VISIBLE_DEVICES='-1', PYTHONIOENCODING='utf-8', RA_DOOR_SKIP='20260918 15:06-15:29',
            RA_DOOR_NOCNN='1', RA_DOOR_NODEPTH='1', RA_DOOR_NOLK='1', RA_DOOR_NOTAP='1')
for name, run in (('micro1s3', 'micro1s3'), ('sam31', 'sam31')):
    for st in ('0', '1'):
        print('==', name, 'stitch', st, flush=True)
        r = subprocess.run([PY, 'door_rule.py', 'fit', '%s_1719_s%s' % (name, st), '20260917=data/door_v2/20260917_%s.jsonl.gz' % run,
                            '20260919=data/door_v2/20260919_%s.jsonl.gz' % run], cwd=str(ROOT), env=dict(base, RA_DOOR_STITCH=st),
                           capture_output=True, text=True)
        print(r.stdout[-1500:], r.stderr[-800:] if r.returncode else '', flush=True)
