import glob, json, os
exec(open('rescore.py').read().split('def drop_gray')[0])
def dg(tag, ev): return [e for e in ev if not any(abs(e['t'] - g) <= 6 for g in gray.get(tag, []))]
for f in sorted(glob.glob('big2_events_*.json')):
    O = json.load(open(f))
    for name, evs in O.items():
        r = {}
        for role in ('train', 'test'):
            S = {'in': [0, 0, 0], 'out': [0, 0, 0]}
            for tag, ev in evs.items():
                if d['tags'][tag]['role'] != role: continue
                m = R.match(dg(tag, ev), dg(tag, truth[tag]))
                for k in S:
                    for i in range(3): S[k][i] += m[k][i]
            r[role] = (round(sum(2 * S[k][0] / max(1, S[k][1] + S[k][2]) for k in S) / 2, 3), S)
        print('%-22s train %.3f  test %.3f  test %s' % (name, r['train'][0], r['test'][0], r['test'][1]))
# reference: live logic on teacher tracks / micro1s3 from zone_baseline, same corrected truth
b = json.load(open('zone_baseline.json'))
for name in ('sam31', 'micro1s3'):
    for role in ('train', 'test'):
        S = {'in': [0, 0, 0], 'out': [0, 0, 0]}
        for tag, v in b.items():
            if v['role'] != role or name not in v or 'error' in v[name]: continue
            m = R.match(dg(tag, v[name]['events']), dg(tag, truth[tag]))
            for k in S:
                for i in range(3): S[k][i] += m[k][i]
        print('live logic on %-9s %s %.3f %s' % (name, role, sum(2 * S[k][0] / max(1, S[k][1] + S[k][2]) for k in S) / 2, S))
