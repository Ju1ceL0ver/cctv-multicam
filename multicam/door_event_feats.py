"""20.10: appearance features of the live door's event views (data/live/event_views/<day>/<key>/), offline.

TransReID clothes + CSCI shape on the processor (3 threads, started below normal priority), cached by the view's
sha256 in data/door_identity/<day>/live_features.npz. Then door_presence links exits to the people inside ->
presence_live.json, and _sheet*.jpg (exit | its entry or the two best candidates) for a quick look.
/visitpairs orders its candidates by these features.

-> data/live/records/identity_<day>.json: per event its visit ID (entry -> its exit, returns), status, role.

usage: door_event_feats.py DAY [--loop UNTIL_HH:MM]   (venv_rfdetr; RA_ID_EVERY seconds between passes, 60)"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))


def embed(day, emb=None):
    import cv2
    base = ROOT / 'data' / 'live' / 'event_views' / day
    out = ROOT / 'data' / 'door_identity' / day
    out.mkdir(parents=True, exist_ok=True)
    cache = out / 'live_features.npz'
    have = dict(np.load(cache)) if cache.exists() else {}
    ev = {}
    for d in sorted(base.iterdir()) if base.exists() else []:
        try:
            m = json.load(open(d / 'manifest.json', encoding='utf-8'))
        except (OSError, ValueError):
            continue
        ev[d.name] = m
        todo = [v for v in m['views'] if 'c_' + v['sha256'] not in have]
        ims = [cv2.imread(str(d / v['path'])) for v in todo]
        todo = [v for v, im in zip(todo, ims) if im is not None]
        ims = [im for im in ims if im is not None]
        if not todo:
            continue
        if emb is None:
            import torch, track_emb
            torch.set_num_threads(int(os.environ.get('RA_ID_THREADS', '3')))
            emb = track_emb.teachers('cpu')
        c, s = emb(ims)
        for v, a, b in zip(todo, c, s):
            have['c_' + v['sha256']] = np.asarray(a, np.float32) / max(float(np.linalg.norm(a)), 1e-9)
            have['s_' + v['sha256']] = np.asarray(b, np.float32) / max(float(np.linalg.norm(b)), 1e-9)
        tmp = out / 'live_features.tmp.npz'
        np.savez(tmp, **have); os.replace(tmp, cache)
    return ev, have, emb


def link(day, ev, have, cloth_weight=.5, sheets=True):
    import cv2
    from door_presence import match
    from storage import atomic_json
    base = ROOT / 'data' / 'live' / 'event_views' / day
    out = ROOT / 'data' / 'door_identity' / day
    w = cloth_weight
    V = {k: np.array([np.r_[np.sqrt(w) * have['c_' + v['sha256']], np.sqrt(1 - w) * have['s_' + v['sha256']]]
                      for v in m['views'] if 'c_' + v['sha256'] in have])
         for k, m in ev.items() if any('c_' + v['sha256'] in have for v in m['views'])}
    roles = {}                                     # door_role's verdict per event key: two inventories, staff and customers
    rp = ROOT / 'data' / 'live' / 'records' / ('roles_%s.jsonl' % day)
    for line in open(rp, encoding='utf-8') if rp.exists() else []:
        try:
            x = json.loads(line); roles[x['key']] = x['role'] if x.get('role') in ('staff', 'customer') else None
        except (ValueError, KeyError):
            continue
    events = [dict(key=k, t=m['event']['t'], kind=m['event']['kind'], clock=m['event']['clock'], w=m['event'].get('w'),
                   role=roles.get(k)) for k, m in ev.items()]
    r = match(events, V)
    atomic_json(out / 'presence_live.json', r)
    # 20.10: one visit ID per event for the rest of the system (CRM untouched). Decisions are chronological, so a
    # published line does not change later; only exits within GROUP_S of the newest events are still open.
    import door_presence as DP
    t_of = {e['key']: e for e in events}
    newest = max((e['t'] for e in events), default=0)
    # 21.10 owner: one entry and one exit per visit, however long one lingers in the doorway. count: the visit's first
    # entry; its last exit once RETURN_S has passed with no return (None = not decided yet); an exit nobody matched.
    first_in, last_out = {}, {}
    for x in r:
        if x['person'] and t_of[x['event']]['kind'] == 'in':
            first_in.setdefault(x['person'], x['event'])
        if x['person'] and t_of[x['event']]['kind'] == 'out':
            last_out[x['person']] = x['event']
    def count(x):
        e = t_of[x['event']]
        if e.get('role') == 'staff':             # 22.10 owner: every staff crossing counts (out for a smoke and back in)
            return x['status'] not in ('repeat_entry', 'repeat_exit')
        if e['kind'] == 'in':
            return first_in.get(x['person']) == x['event']
        if not x['person']:
            return True
        if last_out.get(x['person']) != x['event']:
            return False
        return True if e['t'] < time.time() - DP.RETURN_S else None
    lines = [dict(key=x['event'], kind=t_of[x['event']]['kind'], clock=t_of[x['event']]['clock'], count=count(x),
                  role=t_of[x['event']].get('role'), visit=x['person'], entry=x['entry_event'], status=x['status'],
                  score=round(x['candidates'][0]['score'], 3) if x.get('candidates') else None,
                  final=t_of[x['event']]['t'] < newest - DP.GROUP_S)
             for x in r]
    atomic_json(ROOT / 'data' / 'live' / 'records' / ('identity_%s.json' % day), lines)

    if not sheets:
        return r

    def img(k):
        if k is None or not ev.get(k, {}).get('views'):
            return np.zeros((240, 120, 3), np.uint8)
        im = cv2.imread(str(base / k / ev[k]['views'][0]['path']))
        return np.zeros((240, 120, 3), np.uint8) if im is None else cv2.resize(im, (120, 240))
    rows = []
    for x in r:
        if not x['event'].startswith('out'):
            continue
        c = [x['entry_event']] if x['entry_event'] else [c['entry_event'] for c in x['candidates'][:2]]
        c = (c + [None, None])[:2]
        im = np.hstack([img(x['event'])] + [img(k) for k in c])
        cv2.putText(im, '%s %s %s' % (x['event'][4:10], x['status'][:5], ','.join('%.2f' % q['score'] for q in x['candidates'][:2])),
                    (2, 14), 0, .42, (0, 255, 255), 1)
        rows.append(im)
    for p in out.glob('_sheet*.jpg'):
        p.unlink()
    for i in range(0, len(rows), 12):
        g = rows[i:i + 12]; g += [np.zeros_like(rows[0])] * (-len(g) % 3)
        cv2.imwrite(str(out / ('_sheet%d.jpg' % (i // 12))), np.vstack([np.hstack(g[j:j + 3]) for j in range(0, len(g), 3)]))
    return r


if __name__ == '__main__':
    a = sys.argv[1:]
    day = a[0]
    until = a[a.index('--loop') + 1] if '--loop' in a else None
    emb = None
    while True:
        ev, have, emb = embed(day, emb)
        last = until is None or time.strftime('%H:%M') >= until
        r = link(day, ev, have, sheets=last)
        print(time.strftime('%H:%M:%S'), 'events', len(ev), 'matched', sum(x['status'] == 'matched' for x in r),
              'pending', sum(x['status'] == 'pending' for x in r), flush=True)
        if last:
            break
        time.sleep(float(os.environ.get('RA_ID_EVERY', '60')))
