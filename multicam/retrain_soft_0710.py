"""07.10: after retrain_0710 -> frames again (with the time to the nearest change), the soft refit (v3), the fair frame
test, then v3 on the test stretches' tracks and entries/exits. Log: data/logs/retrain.log"""
import json, os, subprocess, sys, time
from pathlib import Path
import psutil
R = Path(__file__).resolve().parent
sys.path.insert(0, str(R))
import door_side as S
log = open(R / 'data' / 'logs' / 'retrain.log', 'a', encoding='utf-8')
say = lambda m: (log.write(time.strftime('%m-%d %H:%M:%S ') + m + '\n'), log.flush())
while any('retrain_0710.py' in ' '.join(p.info['cmdline'] or []) for p in psutil.process_iter(['cmdline'])):
    time.sleep(30)
env = dict(os.environ, PYTHONIOENCODING='utf-8')
run = lambda args, extra=None: subprocess.run([sys.executable] + args, cwd=R, env=dict(env, **(extra or {})), stdout=log, stderr=log)
say('soft: collect again')
run(['doorside_train.py', 'collect', '20260917'])
run(['doorside_train.py', 'fit_soft', '20260918', '20260919'])
tests = {d: [t for t, s in S.load(d).items() if s.get('done') and s.get('role') == 'test'] for d in ('20260918', '20260919')}
ps = [subprocess.Popen([sys.executable, 'door_binary_count.py', 'predict', d], cwd=R,
                       env=dict(env, RA_BIN_SRC=src, RA_BIN_VER='v3_', RA_BIN_MODEL=str(R / 'inout_lab' / 'binary_door_v3.pkl'),
                                RA_BIN_TAGS=','.join(tags), OMP_NUM_THREADS='4'), stdout=log, stderr=log)
      for src in ('micro1s3', 'sam31') for d, tags in tests.items() if tags]
for p in ps:
    p.wait()
say('entries/exits, model v3')
run(['doorside_eval.py'], {'RA_BIN_VER': 'v3_', 'RA_SIDE_ROLE': 'test'})
say('soft done')
