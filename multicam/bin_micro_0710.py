"""07.10: the owner's binary model on the small SAM's door tracks (micro1s3) of 17-19.09: predictions (three days at
once), positions, the door-zone gated score -> data/door_v2/binary_gated_micro1s3.json."""
import os, subprocess, sys
from pathlib import Path
R = Path(__file__).resolve().parent
env = dict(os.environ, PYTHONIOENCODING='utf-8', RA_BIN_SRC='micro1s3', OMP_NUM_THREADS='4')
days = ('20260917', '20260918', '20260919')
for step in ('predict', 'positions'):
    ps = [subprocess.Popen([sys.executable, 'door_binary_count.py', step, d], cwd=R, env=env,
                           stdout=open(R / 'data' / 'logs' / ('binmicro_%s_%s.log' % (step, d)), 'w'), stderr=subprocess.STDOUT) for d in days]
    for p in ps:
        p.wait()
subprocess.run([sys.executable, 'door_binary_count.py', 'gated'], cwd=R, env=env)
