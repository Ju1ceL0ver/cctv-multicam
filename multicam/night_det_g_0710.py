import os, subprocess, sys
from pathlib import Path
R = Path(__file__).resolve().parent
PY = r'C:\Users\ArykovAA\cctv_ai\venv_sam3\Scripts\python.exe'
e = R / 'runs' / 's31e2e_e'
subprocess.run([PY, 'sam31_e2e_det.py', 's31e2e_g', str(e / 'last.pt'), '09:30'], cwd=R,
               env=dict(os.environ, PYTHONIOENCODING='utf-8', RA_E2E_PROMPT='1', RA_E2E_INIT_HEADS=str(e / 'heads.pt'), RA_E2E_LR_SCALE='0.5'),
               stdout=open(R / 'data' / 'logs' / 'e2edet_g.log', 'w'), stderr=subprocess.STDOUT)
