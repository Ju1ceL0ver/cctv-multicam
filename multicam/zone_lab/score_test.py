import glob, json
exec(open('rescore.py').read().split('def drop_gray')[0])
def dg(tag, ev): return [e for e in ev if not any(abs(e['t'] - g) <= 6 for g in gray.get(tag, []))]
import door_side as S
done = {}
for f in glob.glob('../data/door_side/2026*.json'):
    if 'history' in f: continue
    for tag, st in json.load(open(f, encoding='utf-8')).items():
        if isinstance(st, dict): done[tag] = (st.get('role', 'test'), bool(st.get('done')))
b = json.load(open('zone_baseline.json'))
O = {}
for f in glob.glob('big2_events_*.json'):
    O.update(json.load(open(f)))
tags = sorted(t for t, (role, dn) in done.items() if role != 'train' and dn and t in b and 'micro1s3' in b[t] and 'error' not in b[t]['micro1s3']
              and all(t in ev for ev in O.values()))
print('test stretches, labelled done, present everywhere:', len(tags), tags)
def score(get):
    Sx = {'in': [0, 0, 0], 'out': [0, 0, 0]}
    for t in tags:
        m = R.match(dg(t, get(t)), dg(t, truth[t]))
        for k in Sx:
            for i in range(3): Sx[k][i] += m[k][i]
    return round(sum(2 * Sx[k][0] / max(1, Sx[k][1] + Sx[k][2]) for k in Sx) / 2, 3), Sx
print('%-24s %s' % ('live: micro1s3', score(lambda t: b[t]['micro1s3']['events'])))
print('%-24s %s' % ('live logic: sam31', score(lambda t: b[t]['sam31']['events'])))
for name in sorted(O):
    print('%-24s %s' % (name, score(lambda t: O[name][t])))
