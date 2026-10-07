"""07.10 evening: the owner's /doorside stretches of 17.09 -> TRAIN, 18-19.09 -> TEST; frames of the train stretches
-> refit of the inside/outside model (binary_door_v2.pkl); the test stretches scored per frame (old vs v2) and as
entries/exits on the small SAM's and SAM 3.1's tracks (old vs v2). Log: data/logs/retrain.log"""
import json, os, subprocess, sys, time
from pathlib import Path
R = Path(__file__).resolve().parent
sys.path.insert(0, str(R))
import door_side as S
log = open(R / 'data' / 'logs' / 'retrain.log', 'a', encoding='utf-8')
say = lambda m: (log.write(time.strftime('%m-%d %H:%M:%S ') + m + '\n'), log.flush())
env = dict(os.environ, PYTHONIOENCODING='utf-8')
for day, role in (('20260917', 'train'), ('20260918', 'test'), ('20260919', 'test')):
    for tag, s in S.load(day).items():
        if s.get('done'):
            S.change(day, {'tag': tag, 'act': 'role', 'role': role})
say('roles set: 17.09 train, 18-19.09 test')
run = lambda args, extra=None: subprocess.run([sys.executable] + args, cwd=R, env=dict(env, **(extra or {})), stdout=log, stderr=log)
run(['doorside_train.py', 'collect', '20260917'])
run(['doorside_train.py', 'fit', '20260918', '20260919'])
tests = {d: [t for t, s in S.load(d).items() if s.get('done') and s.get('role') == 'test'] for d in ('20260918', '20260919')}
say('test stretches %s' % json.dumps(tests))
ps = []
for src in ('micro1s3', 'sam31'):
    for d, tags in tests.items():
        if tags:
            ps.append(subprocess.Popen([sys.executable, 'door_binary_count.py', 'predict', d], cwd=R,
                                       env=dict(env, RA_BIN_SRC=src, RA_BIN_VER='v2_', RA_BIN_MODEL=str(R / 'inout_lab' / 'binary_door_v2.pkl'),
                                                RA_BIN_TAGS=','.join(tags), OMP_NUM_THREADS='4'), stdout=log, stderr=log))
for p in ps:
    p.wait()
for ver in ('', 'v2_'):
    say('entries/exits, model %s' % (ver or 'old'))
    run(['doorside_eval.py'], {'RA_BIN_VER': ver, 'RA_SIDE_ROLE': 'test'})
say('done')
