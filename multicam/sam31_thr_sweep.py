"""SAM 3.1's new-person / detection thresholds for the stage-1 student (06.10.2026): the full pipeline against the
cached teacher on 23.09, one setting after another."""
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PY = str(ROOT.parent / 'venv_sam3' / 'Scripts' / 'python.exe')
ck = sys.argv[1] if len(sys.argv) > 1 else 'runs/s31micro_a/last.pt'
out = {}
for nd, sc in ((0.65, 0.4), (0.5, 0.3), (0.4, 0.25), (0.3, 0.2)):
    o = ROOT / 'data' / 'logs' / ('thr_%s_%s.json' % (nd, sc))
    subprocess.run([PY, 'sam31_lite_eval.py', ck, '2', '96', '--cache', 'runs/s31micro_a/evals/teacher_cache', '--out', str(o),
                    '--preview', str(o.with_suffix('.jpg')), '--film', 'none'], cwd=str(ROOT),
                   env=dict(os.environ, RA_S31_NEWDET=str(nd), RA_S31_SCORE=str(sc), PYTHONIOENCODING='utf-8'), capture_output=True)
    out['%s/%s' % (nd, sc)] = json.load(open(o))['mean'] if o.exists() else 'failed'
    print(nd, sc, out['%s/%s' % (nd, sc)], flush=True)
json.dump(out, open(ROOT / 'data' / 'logs' / 'thr_sweep.json', 'w'), indent=1)
