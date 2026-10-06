"""How the live door would look (06.10.2026): the small SAM's tracks of a door stretch, left as they come, right as
the live system would show them -- with DELAY seconds of delay, so that at every moment the broken tracks are
stitched (door_stitch: duplicates dropped, a track starting where another just ended takes its number) using only
what is seen up to DELAY s later; the numbers shown never change afterwards. Entries and exits of the door rule
(fitted on stitched tracks) and the owner's truth at the bottom.

usage: door_live_video.py MICRO_NAME TAG [DELAY] [RULE]  -> data/logs/door_live_<tag>.mp4 (CPU)"""
import copy
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))


def main(name, tag, delay=10, rule_name='micro1s3_1719_s1'):
    import cv2
    import day_proxy
    import door_compare_video as V
    import door_learn as L
    import door_rule as DR
    import door_stitch as ST
    import door_v2 as D
    delay = float(delay)
    mb = ROOT / 'data' / 'micro_door' / name / tag / 'cam1'
    tb = ROOT / 'data' / 'sam31_door' / tag / 'cam1'
    M, by, info = V.people(mb)
    stride = int(info.get('stride', 1))
    tick = D.TICK * stride
    day = tag.split('_')[1]
    a = next((x for x, y in D.stretches(day) if int(round(x)) == int(tag.split('_')[2])), float(tag.split('_')[2]))
    ticks = V.ticks_of(M, by, a, tick)
    row_of = {}                                          # (t, w) -> row, to draw the masks of the stitched people
    for k, items in by.items():
        for r, pid in items:
            row_of[(round(a + k * tick, 2), pid)] = r
    for rr in ticks:
        for q in rr['p']:
            q['r'] = row_of[(rr['t'], q['w'])]
    # the live view: at each moment, stitching that knows DELAY seconds ahead
    shown = {}
    times = [rr['t'] for rr in ticks]
    for i, t in enumerate(times):
        win = [copy.deepcopy(x) for x in ticks if x['t'] <= t + delay]
        ST.stitch(win)
        now = next(x for x in win if x['t'] == t)
        shown[t] = [(q['r'], q['w']) for q in now['p']]
    # events: the rule on the whole stitched stretch (live tells each one LAG s after it, at its crossing time)
    full = copy.deepcopy(ticks)
    ST.stitch(full)
    D.add_io(full)
    rule = DR.load(rule_name)
    ev = DR.apply(full, rule)
    n_t = int(json.load(open(tb / 'info.json'))['ticks'])
    owner = [{'kind': x['kind'], 't': x['t'] - L.SHIFT} for x in D.truth(day, True)
             if x['kind'] in ('in', 'out') and a - 1 <= x['t'] - L.SHIFT <= a + n_t * D.TICK]
    raw_ids = len({w for x in ticks for w in [q['w'] for q in x['p']]})
    live_ids = len({w for v in shown.values() for _, w in v})
    print(json.dumps({'tag': tag, 'people raw': raw_ids, 'people live': live_ids,
                      'events': ', '.join('%s %.1f' % (e['kind'], e['t'] - a) for e in ev),
                      'owner': ', '.join('%s %.1f' % (e['kind'], e['t'] - a) for e in owner)}), flush=True)
    out = str(ROOT / 'data' / 'logs' / ('door_live_%s.mp4' % tag))
    Wd, Hd = 960, 540
    enc = subprocess.Popen([day_proxy._ffmpeg(), '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-s', '%dx%d' % (2 * Wd, Hd),
                            '-r', str(12.5 / stride), '-i', '-', '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '23', '-pix_fmt', 'yuv420p', out],
                           stdin=subprocess.PIPE)
    cap = cv2.VideoCapture(str(tb / 'video.mp4'))
    k = 0
    while True:
        ok, f = cap.read()
        if not ok:
            break
        if k % stride == 0:
            t = round(a + (k // stride) * tick, 2)
            A = V.draw(f.copy(), M, by.get(k // stride, []), 'small SAM, raw tracks  t=%.1f' % (t - a))
            B = V.draw(f.copy(), M, shown.get(t, []), 'LIVE: small SAM + stitching, %.0f s delay' % delay)
            V.banner(B, ev, t, 'RULE', 20, 1)
            V.banner(A, owner, t, 'OWNER', 20, 2)
            V.banner(B, owner, t, 'OWNER', 20, 2)
            enc.stdin.write(np.hstack([cv2.resize(A, (Wd, Hd)), cv2.resize(B, (Wd, Hd))]).tobytes())
        k += 1
    enc.stdin.close(); enc.wait()
    print(out, flush=True)


if __name__ == '__main__':
    main(*sys.argv[1:])
