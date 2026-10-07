"""07.10: the live door (door_live.py, rule + the owner's model on agreement) on the N stretches of 19.09 with the most
of the owner's crossings, fed as files at the record's own clock; its events against the truth, and the offline
combination on the same stretches. -> data/live/test_0710.json, log data/logs/livetest.log"""
import json, os, shutil, subprocess, sys, time
from pathlib import Path
R = Path(__file__).resolve().parent
sys.path.insert(0, str(R))
PY = r'C:\Users\ArykovAA\cctv_ai\venv_sam3\Scripts\python.exe'
N = int(sys.argv[1]) if len(sys.argv) > 1 else 4
import door_learn as L
import door_v2 as D
day = '20260919'
truth = D.truth(day, True)
stretches = []
for tag in sorted(p.parent.name for p in (R / 'data' / 'sam31_door').glob('door_%s_*/cam1' % day)):
    a = float(tag.split('_')[2])
    a = next((x for x, y in D.stretches(day) if int(round(x)) == int(a)), a)
    info = json.load(open(R / 'data' / 'sam31_door' / tag / 'cam1' / 'info.json'))
    b = a + int(info['ticks']) * 0.08
    n = sum(a + 5 <= t['t'] - L.SHIFT <= b - 25 for t in truth if t['kind'] in ('in', 'out'))
    stretches.append((n, tag, a, b))
stretches.sort(reverse=True)
pick = stretches[:N]
ev_path = R / 'data' / 'live' / 'events_cam1.jsonl'
allev = []
for n, tag, a, b in pick:
    if ev_path.exists():
        ev_path.unlink()
    shutil.rmtree(R / 'data' / 'live' / 'cam1', ignore_errors=True)
    t0 = time.time()
    subprocess.run([PY, 'door_live.py', r'C:\Users\ArykovAA\cctv_ai\multicam\runs\s31micro_a\last.pt', '-',
                    'micro1s3_live3', '--source', str(R / 'data' / 'sam31_door' / tag / 'cam1' / 'video.mp4'), '--stride', '3'],
                   cwd=R, env=dict(os.environ, PYTHONIOENCODING='utf-8', RA_SOURCE_T0=str(a)), capture_output=True)
    ev = [json.loads(l) for l in open(ev_path)] if ev_path.exists() else []
    allev += [dict(e, tag=tag) for e in ev]
    print(tag, 'truth', n, 'events', len(ev), '%.0f s' % (time.time() - t0), flush=True)
spans = [(a + 5, b - 25) for n, tag, a, b in pick]
inside = lambda t: any(x <= t <= y for x, y in spans)
T = [t for t in truth if inside(t['t'] - L.SHIFT)]
live = [dict(e, t=e['t'] + L.SHIFT) for e in allev if inside(e['t'])]
m = D.match(live, T, L.TOL)
rep = {'stretches': [p[1] for p in pick], 'truth': len(T), 'live': {k: m[k] for k in ('in', 'out')},
       'live_f1': round((m['in']['f1'] + m['out']['f1']) / 2, 3)}
json.dump(rep, open(R / 'data' / 'live' / 'test_0710.json', 'w'), indent=1)
print(json.dumps(rep), flush=True)
