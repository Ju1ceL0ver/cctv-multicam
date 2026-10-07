import os, subprocess, sys
from pathlib import Path
R = Path(__file__).resolve().parent
for src in ('micro1s3', 'sam31'):
    env = dict(os.environ, PYTHONIOENCODING='utf-8', RA_BIN_SRC=src)
    if not (R / 'data' / 'door_v2' / ('learn_preds_%s_20260919.json' % src)).exists():
        subprocess.run([sys.executable, 'door_combo.py', 'rules'], cwd=R, env=env)
    subprocess.run([sys.executable, 'door_combo.py', 'eval'], cwd=R, env=env)
