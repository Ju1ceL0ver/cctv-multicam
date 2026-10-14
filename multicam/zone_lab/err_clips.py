"""15.10 owner: every remaining error of the live door setting on the /doorside truth, as a short video.
For each missed and each false crossing: +-4 s of the stretch, every track's mask with its number and side
(green inside / magenta outside / grey undecided), the truth and the system's crossings written on top.
-> ../data/door_v2/err_live/<n>.mp4 + index.json (tag, role, t, kind, what: missed|false, people near the door).
usage (from zone_lab, ZL_CACHE set): err_clips.py [train|test|all]"""
import json, os, subprocess, sys
import numpy as np, cv2
src = open('exp_queue.py').read().split('done = set()')[0]
exec(src)
from door_seg_track import Config as TC
ROLE = sys.argv[1] if len(sys.argv) > 1 else 'train'
TR = {'max_gap': 15, 'cont': .5}
CFG = dict(Q.BASE4, SM=0., thr=.6, inh_overlap=.3, door_s=2., pair_s=3., rules_first=1)
OUT = os.path.join('..', 'data', 'door_v2', 'err_live'); os.makedirs(OUT, exist_ok=True)
FF = os.environ.get('RA_FFMPEG') or r'C:\Users\ArykovAA\AppData\Local\miniconda3\envs\cctv_base\Lib\site-packages\imageio_ffmpeg\binaries\ffmpeg-win-x86_64-v7.1.exe'
FPS, W, H = 12.5, 960, 540
index, n = [], 0
for t in TAGS:
    role = d['tags'][t]['role']
    if ROLE != 'all' and role != ROLE:
        continue
    by = L.load(t)
    tracks = L.track(by, TC(max_gap=TR['max_gap']), birth=.4, cont=TR['cont'])
    rows = L.rows_of(tracks)
    ev, sides = Q.events(rows, CFG)
    ev, tr = dg(t, ev), dg(t, truth[t])
    errs = [('missed', x) for x in tr if not any(abs(e['t'] - x['t']) <= 6 and e['kind'] == x['kind'] for e in ev)]
    errs += [('false', e) for e in ev if not any(abs(e['t'] - x['t']) <= 6 and e['kind'] == x['kind'] for x in tr)]
    if not errs:
        continue
    cap = cv2.VideoCapture(os.path.join('..', 'data', 'sam31_door', t, 'cam1', 'video.mp4'))
    frames = []
    while True:
        ok, f = cap.read()
        if not ok: break
        frames.append(cv2.resize(f, (W, H), interpolation=cv2.INTER_AREA))
    cap.release()
    last_side = {}
    for what, e in sorted(errs, key=lambda x: x[1]['t']):
        k0, k1 = max(0, int((e['t'] - 4) * FPS)), min(len(frames) - 1, int((e['t'] + 4) * FPS))
        name = '%03d' % n; n += 1
        p = subprocess.Popen([FF, '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-s', '%dx%d' % (W, H), '-r', str(FPS),
                              '-i', '-', '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '25', '-pix_fmt', 'yuv420p', '-movflags', '+faststart',
                              os.path.join(OUT, name + '.mp4')], stdin=subprocess.PIPE)
        near = set()
        for k in range(k0, k1 + 1):
            img = frames[k].copy()
            for pid, pr in tracks.get(k, []):
                m = pr[0][:180]
                if not m.any(): continue
                s = None
                for kk in (k, k - 1, k - 2):
                    if (str(pid), kk) in sides: s = sides[(str(pid), kk)]; break
                if s is None: s = last_side.get(str(pid))
                last_side[str(pid)] = s
                col = (60, 200, 60) if s == 1 else (200, 60, 200) if s == 0 else (170, 170, 170)
                big = cv2.resize(m.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST)
                img[big > 0] = (0.55 * img[big > 0] + 0.45 * np.array(col)).astype(np.uint8)
                cn, _ = cv2.findContours(big, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(img, cn, -1, col, 2)
                ys, xs = np.nonzero(big)
                cv2.putText(img, '#%s' % pid, (int(xs.min()), max(14, int(ys.min()) - 4)), cv2.FONT_HERSHEY_SIMPLEX, .55, col, 2)
                if abs(k / FPS - e['t']) < 1.5: near.add(str(pid))
            cv2.rectangle(img, (0, 0), (W, 50), (0, 0, 0), -1)
            head = '%s %s  %s  t=%.1f s  (%+.1f)' % ('MISSED' if what == 'missed' else 'FALSE', e['kind'].upper(), t, e['t'], k / FPS - e['t'])
            cv2.putText(img, head, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, .6, (0, 0, 255) if what == 'false' else (0, 200, 255), 2)
            tt = ' '.join('%s%.0f' % (x['kind'][0], x['t']) for x in tr if abs(x['t'] - e['t']) < 8)
            ss = ' '.join('%s%.0f#%s' % (x['kind'][0], x['t'], x['w']) for x in ev if abs(x['t'] - e['t']) < 8)
            cv2.putText(img, 'truth: %s   system: %s' % (tt, ss), (8, 42), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), 1)
            p.stdin.write(img.tobytes())
        p.stdin.close(); p.wait()
        index.append({'n': name, 'tag': t, 'role': role, 't': round(e['t'], 1), 'kind': e['kind'], 'what': what, 'near': sorted(near)})
        print(name, what, e['kind'], t, round(e['t'], 1), flush=True)
json.dump(index, open(os.path.join(OUT, 'index.json'), 'w'), indent=0)
print('errors', len(index))
