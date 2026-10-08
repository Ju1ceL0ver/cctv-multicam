"""Short clips of the live door for labelling and checking (08.10.2026) -- instead of the whole raw recording.

From the frames the live window keeps anyway (frames/<k>.jpg, 2176 x 1224, one per tick), 1280 x 720 H.264, nothing
drawn on them; next to each clip a json with every frame's clock time and why it was kept:
  event   -- +-PAD s around every counted entry/exit (its kind, track, role)
  dispute -- a track whose side votes changed but nothing was counted, or the model and the line disagreed for a while
  random  -- one random minute per hour (what the system did not pick: misses)
At night SAM 3.1 labels these clips (the teacher), the same door logic counts on its tracks, and where the teacher and
the live system disagree goes to the owner (/doorside); where they agree is training data as it is.
-> data/live/clips/<day>/<HHMMSS>_<why>.mp4 + .json. Overlapping requests of one window are merged into one clip."""
import glob
import json
import os
import random
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'data' / 'live' / 'clips'
PAD = 10.0
DISPUTE_TICKS = 3               # model and line on opposite sides this many ticks of one track -> a dispute


def ffmpeg():
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        hits = glob.glob(r'C:\Users\ArykovAA\AppData\Local\miniconda3\envs\*\Lib\site-packages\imageio_ffmpeg\binaries\ffmpeg*.exe')
        return hits[0] if hits else 'ffmpeg'


class Clipper:
    def __init__(self, seed=None):
        self.want = []                                  # [t0, t1, why, info]
        self.made = []                                  # (t0, t1) already cut
        self.hours = {}
        self.rng = random.Random(seed)
        self.exe = ffmpeg()
        self.asked = set()                              # tracks already asked as disputes

    def event(self, e):
        self.want.append([e['t'] - PAD, e['t'] + PAD, 'event',
                          {k: e.get(k) for k in ('kind', 'clock', 'role', 'p_staff', 'tracks', 'w', 'bw')}])

    def disputes(self, combo, told_tracks):
        """Tracks with both sides in their votes but no counted event, and tracks where the model and the line disagree."""
        import door_line
        sd = combo.side or {}
        seen = {}
        for key, w in getattr(combo, 'last_person_of', {}).items():
            if key not in combo.raw:
                continue
            t, d, top, hh, bx, p, foot = combo.raw[key]
            v = combo.obs.get(key, (None, 0.5))[1]
            lv = door_line.vote2(sd, sd.get('table'), combo.line, d, top, hh, bx) if combo.line else 0.5
            thr = sd.get('thr', 0.9)
            mv = 0.5 if p is None else (1.0 if p >= thr else (0.0 if p <= 1 - thr else 0.5))
            s = seen.setdefault(w, {'t': [], 'sides': set(), 'clash': 0})
            s['t'].append(t)
            if v in (0.0, 1.0):
                s['sides'].add(v)
            if {lv, mv} == {0.0, 1.0}:
                s['clash'] += 1
        for w, s in seen.items():
            if w in told_tracks or w in self.asked or not s['t']:
                continue
            if len(s['sides']) == 2 or s['clash'] >= DISPUTE_TICKS:
                t0, t1 = min(s['t']), max(s['t'])
                mid = (t0 + t1) / 2
                self.asked.add(w)
                self.want.append([max(t0, mid - PAD), min(t1, mid + PAD), 'dispute',
                                  {'track': w, 'both_sides': len(s['sides']) == 2, 'clash_ticks': s['clash']}])

    def random_minute(self, t_start, t_end):
        """One random minute per clock hour, asked as soon as the window reaches it."""
        h = int(t_start // 3600)
        while h * 3600 <= t_end:
            if h not in self.hours:
                self.hours[h] = h * 3600 + self.rng.uniform(0, 3540)
            m = self.hours[h]
            if m is not None and t_start <= m and m + 60 <= t_end:
                self.want.append([m, m + 60, 'random', {}])
                self.hours[h] = None
            h += 1

    def cut(self, win, final=False):
        """Clips of requests the window's frames fully cover (or, at the end, whatever it has)."""
        if not win.times:
            return []
        ta, tb = win.times[0], win.times[-1]
        keep, todo = [], []
        for w in self.want:
            if w[1] <= tb or final:
                if w[0] >= ta - PAD or final:
                    todo.append(w)
            else:
                keep.append(w)
        self.want = keep
        todo.sort(key=lambda w: (w[0], w[1]))
        merged = []
        for w in todo:                                  # overlapping requests -> one clip with all reasons
            w = [max(w[0], ta), min(w[1], tb), w[2], w[3]]
            if w[1] - w[0] < 1:
                continue
            if merged and w[0] <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], w[1])
                merged[-1][2].append(w[2])
                merged[-1][3].append(dict(w[3], why=w[2]))
            else:
                merged.append([w[0], w[1], [w[2]], [dict(w[3], why=w[2])]])
        made = []
        for t0, t1, whys, infos in merged:
            if any(a <= t0 and t1 <= b for a, b in self.made):
                continue
            ks = [k for k, t in enumerate(win.times) if t0 <= t <= t1]
            if len(ks) < 3:
                continue
            day = time.strftime('%Y%m%d', time.localtime(t0))
            (OUT / day).mkdir(parents=True, exist_ok=True)
            why = 'event' if 'event' in whys else whys[0]
            name = '%s_%s' % (time.strftime('%H%M%S', time.localtime(t0)), why)
            fps = 1.0 / max(1e-3, (win.times[ks[-1]] - win.times[ks[0]]) / max(1, len(ks) - 1))
            cmd = [self.exe, '-y', '-loglevel', 'error', '-framerate', '%.3f' % fps, '-start_number', str(ks[0]),
                   '-i', str(win.dir / 'frames' / '%05d.jpg'), '-frames:v', str(len(ks)), '-vf', 'scale=1280:720',
                   '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '20', '-pix_fmt', 'yuv420p', str(OUT / day / (name + '.mp4'))]
            try:
                subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                 creationflags=getattr(subprocess, 'BELOW_NORMAL_PRIORITY_CLASS', 0))
            except OSError:
                continue
            json.dump({'t0': t0, 't1': t1, 'why': whys, 'info': infos, 'window': win.id, 'ticks': ks,
                       'times': [round(win.times[k], 3) for k in ks], 'size': [1280, 720]},
                      open(OUT / day / (name + '.json'), 'w'))
            self.made.append((t0, t1))
            made.append(name)
        return made
