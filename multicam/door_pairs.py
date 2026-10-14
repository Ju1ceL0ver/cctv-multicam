"""Entry <-> exit of the same person on the live door (camera 1): one finished visit per pair.

Each event keeps up to 8 best views of its tracks (data/live/views_seg/<day>/views.json). Views are embedded
with the role's teachers (TransReID clothes + CSCI shape, on the processor, cached per view), an entry and an
exit are compared by the mean of the best view-to-view similarities, and all pairs of the day are decided at
once (Hungarian): an exit only after its entry, a visit 5 s .. 4 h long, a pair only above SIM_MIN.
usage: door_pairs.py [day]  ->  data/visits/<day>/pairs.json"""
import sys, os, json, time
import numpy as np
from pathlib import Path
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
LIVE = ROOT / 'data' / 'live'

MIN_S, MAX_S = 5.0, 4 * 3600.0
SIM_MIN = 0.55
TOP = 3                       # best view pairs averaged


def events(day):
    out = []
    for line in open(LIVE / 'events_cam1.jsonl', encoding='utf-8'):
        e = json.loads(line)
        if e.get('engine') == 'doorseg_c+zones' and e['clock'].replace('-', '').startswith(day):
            out.append(e)
    return out


def roles(day):
    p = LIVE / 'records' / ('roles_%s.jsonl' % day)
    r = {}
    if p.exists():
        for line in open(p, encoding='utf-8'):
            x = json.loads(line)
            r[round(x['t'], 1)] = x
    return r


def embed_views(day, paths):
    """{view file name: vector}, cached in data/visits/<day>/view_emb.npz."""
    out_dir = ROOT / 'data' / 'visits' / day
    out_dir.mkdir(parents=True, exist_ok=True)
    cache = out_dir / 'view_emb.npz'
    have = dict(np.load(cache)) if cache.exists() else {}
    todo = [p for p in paths if p not in have]
    if todo:
        import cv2, torch
        import track_emb as T
        torch.set_num_threads(int(os.environ.get('RA_PAIR_THREADS', '3')))
        embed = T.teachers('cpu')
        folder = LIVE / 'views_seg' / day
        for k in range(0, len(todo), 16):
            chunk = todo[k:k + 16]
            imgs = [cv2.imread(str(folder / p)) for p in chunk]
            ok = [i for i, im in enumerate(imgs) if im is not None]
            if not ok:
                continue
            a1, a2 = embed([imgs[i] for i in ok])
            a1 = np.asarray(a1, np.float32); a2 = np.asarray(a2, np.float32)
            a1 /= np.maximum(np.linalg.norm(a1, axis=1, keepdims=True), 1e-6)
            a2 /= np.maximum(np.linalg.norm(a2, axis=1, keepdims=True), 1e-6)
            v = np.concatenate([a1, a2], 1) / np.sqrt(2)
            for i, row in zip(ok, v):
                have[chunk[i]] = row
            np.savez(cache, **have)
    return have


def similarity(A, B):
    S = A @ B.T
    best = np.sort(S.ravel())[::-1][:TOP]
    return float(best.mean())


def pair(day):
    from scipy.optimize import linear_sum_assignment
    ev = events(day)
    views = json.load(open(LIVE / 'views_seg' / day / 'views.json'))
    rl = roles(day)
    per = []
    for e in ev:
        vs = sorted((x for t in e['tracks'] for x in views.get(str(t), [])), key=lambda z: -z[0])[:8]
        per.append([v[1] for v in vs])
    vec = embed_views(day, sorted({p for v in per for p in v}))
    V = [np.stack([vec[p] for p in v if p in vec]) if any(p in vec for p in v) else None for v in per]
    ins = [i for i, e in enumerate(ev) if e['kind'] == 'in']
    outs = [i for i, e in enumerate(ev) if e['kind'] == 'out']
    S = np.full((len(ins), len(outs)), -1.0)
    for a, i in enumerate(ins):
        for b, j in enumerate(outs):
            dt = ev[j]['t'] - ev[i]['t']
            if MIN_S <= dt <= MAX_S and V[i] is not None and V[j] is not None:
                S[a, b] = similarity(V[i], V[j])
    cost = np.where(S >= SIM_MIN, 1 - S, 10.0)
    r, c = linear_sum_assignment(cost) if S.size else ([], [])
    used_in, used_out, visits = set(), set(), []
    for a, b in zip(r, c):
        if cost[a, b] >= 10:
            continue
        i, j = ins[a], outs[b]
        used_in.add(i); used_out.add(j)
        # margin: how much better this partner is than the next one for either side
        row = np.sort(S[a])[::-1]; col = np.sort(S[:, b])[::-1]
        margin = min(S[a, b] - (row[1] if len(row) > 1 else -1), S[a, b] - (col[1] if len(col) > 1 else -1))
        visits.append({'in': ev[i]['clock'], 'out': ev[j]['clock'], 't_in': ev[i]['t'], 't_out': ev[j]['t'],
                       'seconds': round(ev[j]['t'] - ev[i]['t'], 1), 'sim': round(float(S[a, b]), 3),
                       'margin': round(float(margin), 3), 'in_tracks': ev[i]['tracks'], 'out_tracks': ev[j]['tracks'],
                       'in_crop': ev[i].get('crop'), 'out_crop': ev[j].get('crop'),
                       'role': (rl.get(round(ev[i]['t'], 1)) or {}).get('role'),
                       'in_views': per[i], 'out_views': per[j]})
    lonely = [{'kind': ev[i]['kind'], 'clock': ev[i]['clock'], 't': ev[i]['t'], 'tracks': ev[i]['tracks'],
               'crop': ev[i].get('crop'), 'views': per[i],
               'role': (rl.get(round(ev[i]['t'], 1)) or {}).get('role')}
              for i in range(len(ev)) if i not in used_in and i not in used_out]
    chosen = {outs[b]: ins[a] for a, b in zip(r, c) if cost[a, b] < 10}
    exits = []                     # one question per exit for the owner: which entry (or none)
    for b, j in enumerate(outs):
        order = [a for a in np.argsort(-S[:, b]) if S[a, b] > -1][:6]
        exits.append({'key': '%.1f' % ev[j]['t'], 'clock': ev[j]['clock'], 't': ev[j]['t'], 'views': per[j],
                      'crop': ev[j].get('crop'), 'photo': ev[j].get('photo'), 'role': (rl.get(round(ev[j]['t'], 1)) or {}).get('role'),
                      'system': ('%.1f' % ev[chosen[j]]['t']) if j in chosen else None,
                      'cands': [{'key': '%.1f' % ev[ins[a]]['t'], 'clock': ev[ins[a]]['clock'], 'sim': round(float(S[a, b]), 3), 'crop': ev[ins[a]].get('crop'), 'photo': ev[ins[a]].get('photo'),
                                 'views': per[ins[a]], 'role': (rl.get(round(ev[ins[a]]['t'], 1)) or {}).get('role')}
                                for a in order]})
    # every event of the day in time order and all event-to-event similarities: the owner names people on this
    timeline = [{'key': '%.1f' % e['t'], 'kind': e['kind'], 'clock': e['clock'], 't': e['t'], 'crop': e.get('crop'), 'photo': e.get('photo'),
                 'views': per[i], 'role': (rl.get(round(e['t'], 1)) or {}).get('role')} for i, e in enumerate(ev)]
    sim_all = [[(round(similarity(V[i], V[j]), 3) if V[i] is not None and V[j] is not None and i != j else None)
                for j in range(len(ev))] for i in range(len(ev))]
    visits.sort(key=lambda v: v['t_in'])
    out = {'day': day, 'made': time.strftime('%Y-%m-%d %H:%M:%S'), 'events': len(ev), 'ins': len(ins), 'outs': len(outs),
           'visits': visits, 'unpaired': lonely, 'exits': exits, 'timeline': timeline, 'sim': sim_all, 'sim_min': SIM_MIN}
    from storage import atomic_json
    atomic_json(ROOT / 'data' / 'visits' / day / 'pairs.json', out)
    return out


if __name__ == '__main__':
    day = sys.argv[1] if len(sys.argv) > 1 else time.strftime('%Y%m%d')
    o = pair(day)
    print('events %d (in %d, out %d) -> visits %d, unpaired %d' % (o['events'], o['ins'], o['outs'], len(o['visits']), len(o['unpaired'])))
    for v in o['visits']:
        print('  %s -> %s  %6.0f s  sim %.2f  margin %+.2f  %s' % (v['in'][11:], v['out'][11:], v['seconds'], v['sim'], v['margin'], v['role']))
