import os, subprocess, sys
from pathlib import Path
R = Path(__file__).resolve().parent
env = dict(os.environ, PYTHONIOENCODING='utf-8', RA_BIN_SRC='micro1s3')
subprocess.run([sys.executable, 'door_combo.py', 'eval'], cwd=R, env=env)
subprocess.run([sys.executable, 'door_combo.py', 'final'], cwd=R, env=env)
