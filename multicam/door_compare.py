"""The door rule over the three days on every track source and feature set -> data/door_v2/compare.out (04.10.2026)."""
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
D = ROOT / 'data' / 'door_v2'
RUNS = {'model': {'20260917': '20260917_v2_m_clips_best_q', '20260918': '20260918_v2_m_clips_best', '20260919': '20260919_v2_m_clips_best_p'},
        'sam': {d: '%s_sam31' % d for d in ('20260917', '20260918', '20260919')}}
SETS = {'base': {'RA_DOOR_NOLK': '1', 'RA_DOOR_NOTAP': '1', 'RA_DOOR_NODEPTH': '1'},
        '+points LK': {'RA_DOOR_NOTAP': '1', 'RA_DOOR_NODEPTH': '1'},
        '+points TAPNext++': {'RA_DOOR_NOLK': '1', 'RA_DOOR_NODEPTH': '1'},
        '+depth': {'RA_DOOR_NOLK': '1', 'RA_DOOR_NOTAP': '1'},
        'all': {}}
with open(D / 'compare.out', 'a', encoding='utf-8') as out:
    print('\n==', time.strftime('%m-%d %H:%M'), file=out, flush=True)
    for src, runs in RUNS.items():
        for name, env in SETS.items():
            for skip in ('', '20260918 15:06-15:29'):
                args = ['%s=%s' % (d, D / (n + '.jsonl.gz')) for d, n in runs.items()]
                r = subprocess.run([sys.executable, 'door_learn.py'] + args, cwd=str(ROOT), capture_output=True, text=True,
                                   env=dict(os.environ, RA_DOOR_SKIP=skip, **env))
                print('--', src, '|', name, '|', 'no family' if skip else 'all', file=out)
                print('\n'.join(l for l in r.stdout.splitlines() if l.startswith('2026')) or r.stderr[-600:], file=out, flush=True)
