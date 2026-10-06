"""A door stretch side by side (06.10.2026): SAM 3.1 (the teacher) on the left, the small SAM (micro run, every
stride-th tick) on the right, each person in its own colour with its number, the poster stand left out on both.

usage: door_compare_video.py MICRO_NAME [TAG|busiest] [OUT.mp4]  -> data/logs/door_cmp_<tag>.mp4 (CPU only)"""
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
COL = [(66, 197, 245), (80, 220, 100), (245, 160, 66), (220, 90, 220), (240, 220, 70), (160, 100, 255), (60, 200, 200),
       (200, 160, 90), (120, 230, 180), (250, 120, 160), (90, 140, 250), (230, 230, 230)]


def people(base):
    import door_sam as DS
    import sam31_reid as R
    info = json.load(open(base / 'info.json'))
    rep = json.load(open(base / 'report.json'))
    M = R.Masks(base / 'chunks.npz')
    owned, _ = R.link_seams(M, {int(s): int(sh) for s, e, sh in info['sessions']})
    person = {int(p): v for p, v in rep['person_of_piece'].items()}
    static = DS.static_people(M, owned, person, info['ticks'])
    by_tick = {}
    for p, rs in owned.items():
        pid = int(person.get(int(p)) or 0)
        if pid in static:
            continue
        for r in rs:
            by_tick.setdefault(int(M.rows[r, 1]), []).append((r, pid))
    return M, by_tick, info


def ticks_of(M, by_tick, a, tick):
    """door_v2's run format for one stretch (t = film seconds), as door_sam.convert writes it."""
    import door_sam as DS
    rows = {}
    for k, items in by_tick.items():
        t = round(a + k * tick, 2)
        for r, pid in items:
            x1, y1, x2, y2 = M.rows[r, 4:8].astype(int)
            f = DS.foot_of(M.crop(r))
            rows.setdefault(t, []).append({'w': pid, 's': round(float(M.rows[r, 3]), 3),
                                           'box': [round((x1 + x2) / 2 / 2176, 4), round((y1 + y2) / 2 / 1248, 4),
                                                   round((x2 - x1) / 2176, 4), round((y2 - y1) / 1248, 4)],
                                           'foot': [x1 + f[0], y1 + f[1]] if f else None, 'new': False})
    return [{'s': 0, 't': t, 'p': rows[t]} for t in sorted(rows)]


SIDE = ['OUT', 'IN', 'DOOR']


def events(M, by_tick, a, tick, rule):
    """(the learned rule's events, the label-only events, {(tick time, person): side})"""
    import door_debounce as DB
    import door_rule as DR
    import door_v2 as D
    ticks = ticks_of(M, by_tick, a, tick)
    D.add_io(ticks)
    by, side = {}, {}
    for r in ticks:
        for q in r['p']:
            by.setdefault(q['w'], []).append((r['t'], q))
            side[(round(r['t'], 2), q['w'])] = SIDE[int(np.argmax(q.get('io_mix', q['io'])))]
    deb = DB.events(by, float(os.environ.get('RA_DEB_W', '3')), float(os.environ.get('RA_DEB_FRAC', '0.7')), True)
    return DR.apply(ticks, rule), deb, side


def banner(img, evs, now, who, x0, row=0):
    import cv2
    on = [e for e in evs if 0 <= now - e['t'] <= 2.5]
    y = img.shape[0] - 30 - 70 * (2 - row)
    if on:
        txt = who + ': ' + ' + '.join(('ENTRY P%d' if e['kind'] == 'in' else 'EXIT P%d') % e['w'] if 'w' in e else
                                      ('ENTRY' if e['kind'] == 'in' else 'EXIT') for e in on)
        col = {'RULE': (80, 255, 80), 'LABELS': (255, 200, 80), 'OWNER': (80, 200, 255)}[who]
        cv2.putText(img, txt, (x0, y), cv2.FONT_HERSHEY_SIMPLEX, 1.6, (0, 0, 0), 9)
        cv2.putText(img, txt, (x0, y), cv2.FONT_HERSHEY_SIMPLEX, 1.6, col, 4)


def draw(img, M, items, title, side=None, now=None):
    import cv2
    over = img.copy()
    for r, pid in items:
        x1, y1, x2, y2 = M.rows[r, 4:8].astype(int)
        m = np.zeros(img.shape[:2], bool)
        m[y1:y2, x1:x2] = M.crop(r)
        col = COL[pid % len(COL)]
        over[m] = col
        cs, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(img, cs, -1, col, 4)
        ly = y1 - 12 if y1 > 60 else y2 + 40
        lab = 'P%d %s' % (pid, side.get((round(now, 2), pid), '') if side is not None else '')
        cv2.putText(img, lab, ((x1 + x2) // 2 - 50, ly), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 0, 0), 8)
        cv2.putText(img, lab, ((x1 + x2) // 2 - 50, ly), cv2.FONT_HERSHEY_SIMPLEX, 1.4, col, 3)
    img = cv2.addWeighted(img, 0.65, over, 0.35, 0)
    cv2.rectangle(img, (0, 0), (img.shape[1], 70), (0, 0, 0), -1)
    cv2.putText(img, title, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
    return img


def main(name, tag='busiest', out=None):
    import cv2
    import day_proxy
    micro = ROOT / 'data' / 'micro_door' / name
    if tag == 'busiest':
        best = None
        for d in micro.glob('door_*/cam1/report.json'):
            n = len(json.load(open(d))['person_of_piece'])
            if best is None or n > best[0]:
                best = (n, d.parent.parent.name)
        tag = best[1]
    tb = ROOT / 'data' / 'sam31_door' / tag / 'cam1'
    mb = micro / tag / 'cam1'
    Mt, bt, _ = people(tb)
    Mm, bm, im = people(mb)
    stride = int(im.get('stride', 1))
    import door_learn as L
    import door_rule as DR
    import door_v2 as D
    day = tag.split('_')[1]
    a = float(tag.split('_')[2])
    a = next((x for x, y in D.stretches(day) if int(round(x)) == int(a)), a)
    rule = DR.load(os.environ.get('RA_DOOR_RULE', 'sam31_live'))
    ev_t, deb_t, side_t = events(Mt, bt, a, D.TICK, rule)
    ev_m, deb_m, side_m = events(Mm, bm, a, D.TICK * stride, rule)
    n_t = int(json.load(open(tb / 'info.json'))['ticks'])
    owner = [{'kind': t['kind'], 't': t['t'] - L.SHIFT} for t in D.truth(day, True)
             if t['kind'] in ('in', 'out') and a - 1 <= t['t'] - L.SHIFT <= a + n_t * D.TICK]
    fmt = lambda evs: ', '.join('%s %.1f' % (e['kind'], e['t'] - a) for e in evs)
    summary = {'tag': tag, 'owner': fmt(owner), 'teacher rule': fmt(ev_t), 'teacher labels': fmt(deb_t),
               'small rule': fmt(ev_m), 'small labels': fmt(deb_m)}
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    out = out or str(ROOT / 'data' / 'logs' / ('door_cmp_%s.mp4' % tag))
    Wd, Hd = 960, 540
    enc = subprocess.Popen([day_proxy._ffmpeg(), '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-s', '%dx%d' % (2 * Wd, Hd),
                            '-r', str(12.5 / stride), '-i', '-', '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '23', '-pix_fmt', 'yuv420p', out],
                           stdin=subprocess.PIPE)
    cap = cv2.VideoCapture(str(tb / 'video.mp4'))
    a_s = a
    k = 0
    while True:
        ok, f = cap.read()
        if not ok:
            break
        if k % stride == 0:
            now = a_s + k * D.TICK
            A = draw(f.copy(), Mt, bt.get(k, []), 'SAM 3.1 (teacher)  t=%.1f' % (k * D.TICK), side_t, now)
            B = draw(f.copy(), Mm, bm.get(k // stride, []), 'small SAM (MobileNetV4 encoder)', side_m, now)
            for img, ev, deb in ((A, ev_t, deb_t), (B, ev_m, deb_m)):
                banner(img, ev, now, 'RULE', 20, 0)
                banner(img, deb, now, 'LABELS', 20, 1)
                banner(img, owner, now, 'OWNER', 20, 2)
            enc.stdin.write(np.hstack([cv2.resize(A, (Wd, Hd)), cv2.resize(B, (Wd, Hd))]).tobytes())
        k += 1
    enc.stdin.close(); enc.wait()
    print(out, tag, flush=True)


if __name__ == '__main__':
    main(*sys.argv[1:])
