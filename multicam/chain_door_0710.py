"""07.10.2026 night chain: wait the micro door run -> door rule on 3 days -> comparison videos -> door-only distillation."""
import os, subprocess, sys, time
from pathlib import Path
R = Path(__file__).resolve().parent
PY = sys.executable
SAM = r'C:\Users\ArykovAA\cctv_ai\venv_sam3\Scripts\python.exe'
LOG = open(R / 'data' / 'logs' / 'chain0710.log', 'a', encoding='utf-8')
env = dict(os.environ, PYTHONIOENCODING='utf-8', RA_DOOR_NOCNN='1', RA_DOOR_NODEPTH='1', RA_DOOR_NOLK='1', RA_DOOR_NOTAP='1')

def say(m):
    LOG.write(time.strftime('%m-%d %H:%M:%S ') + m + '\n'); LOG.flush()

def run(args, py=PY, extra=None):
    say('start ' + ' '.join(args))
    r = subprocess.run([py] + args, cwd=R, env=dict(env, **(extra or {})), stdout=LOG, stderr=LOG)
    say('end %d' % r.returncode)

import psutil
while any('door_micro.py' in ' '.join(p.info['cmdline'] or []) for p in psutil.process_iter(['cmdline'])):
    time.sleep(60)
say('micro run finished')
D = R / 'data' / 'door_v2'
for src in ('micro1s3', 'sam31'):
    run(['door_rule.py', 'fit', src + '_live3'] + ['%s=%s' % (d, D / ('%s_%s.jsonl.gz' % (d, src))) for d in ('20260917', '20260918', '20260919')])
for tag in ('busiest',):
    run(['door_compare_video.py', 'micro1s3', tag, str(R / 'data' / 'logs' / 'door_cmp_0710.mp4')])
run(['sam31_distill.py', 's31micro_door', '08:30', '--init', str(R / 'runs' / 's31micro_a' / 'last.pt'),
     '--student', 'mobilenetv4_conv_medium'], py=SAM,
    extra={'RA_DISTILL_KINDS': 'sam31_door', 'RA_DISTILL_SKIP': '20260918'})
say('done')
