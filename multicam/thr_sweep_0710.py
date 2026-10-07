"""07.10: the night detector (s31e2e_b best) on 23.09 at lower thresholds for a new person, then the morning chain."""
import json, os, subprocess, sys, time
from pathlib import Path
R = Path(__file__).resolve().parent
PY = r'C:\Users\ArykovAA\cctv_ai\venv_sam3\Scripts\python.exe'
b = R / 'runs' / 's31e2e_b'
log = open(R / 'data' / 'logs' / 'thrsweep.log', 'a', encoding='utf-8')
for newdet, score in (('0.5', None), ('0.4', None), ('0.4', '0.3')):
    out = b / 'evals' / ('thr_new%s_score%s.json' % (newdet, score or 'def'))
    env = dict(os.environ, PYTHONIOENCODING='utf-8', RA_S31_NEWDET=newdet)
    if score:
        env['RA_S31_SCORE'] = score
    subprocess.run([PY, 'sam31_lite_eval.py', str(b / 'best_last.pt'), '2', '96', '--cache',
                    str(R / 'runs' / 's31micro_a' / 'evals' / 'teacher_cache'), '--out', str(out), '--film', 'none',
                    '--heads', str(b / 'best_heads.pt')], cwd=R, env=env, capture_output=True)
    m = json.load(open(out))['mean'] if out.exists() else 'error'
    log.write('new %s score %s: %s\n' % (newdet, score or 'default', json.dumps(m))); log.flush()
log.write('done\n'); log.flush()
subprocess.Popen([sys.executable, 'chain_0710_morning.py'], cwd=R, env=dict(os.environ, PYTHONIOENCODING='utf-8'),
                 stdout=open(R / 'data' / 'logs' / 'chainmorning.log', 'a'), stderr=subprocess.STDOUT)
