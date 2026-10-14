import json, rules as R
d = R.load('zone_feats.json.gz'); b = json.load(open('zone_baseline.json'))
A = json.load(open('answers.json')); items = json.load(open('items.json'))
c = dict(R.BASE, shop=1, first=0, occl_legs=.6, occl_h=.6, conf=1, pair_s=3)
truth = {t: [dict(e) for e in v['events']] for t, v in d['tags'].items()}
gray = {}                                   # tag -> [(kind or None, t)]
for it in items:
    v = A.get(it['id'], {}).get('v')
    tag, t, kind = it['tag'], it['t'], it['kind']
    T = truth[tag]
    if v == 3 or v == 0:
        gray.setdefault(tag, []).append(t)
        T[:] = [e for e in T if abs(e['t'] - t) > 1.0 or (kind.startswith('extra') and False)]
        if kind.startswith('missed'):
            T[:] = [e for e in T if not (abs(e['t'] - t) < .05 and e['kind'] == kind.split('_')[1])]
        if kind == 'pair':
            T[:] = [e for e in T if abs(e['t'] - t) > .5]
    elif kind == 'pair' and v == 2:
        T[:] = [e for e in T if not (abs(e['t'] - t) < .5 and str(e['person']) in it['note'])]
    elif kind.startswith('missed') and v == 2:
        T[:] = [e for e in T if not (abs(e['t'] - t) < .05 and e['kind'] == kind.split('_')[1])]
    elif kind.startswith('extra') and v == 1:
        T.append({'kind': kind.split('_')[1], 't': t, 'person': 'added'})
def drop_gray(tag, ev):
    return [e for e in ev if not any(abs(e['t'] - g) <= 6 for g in gray.get(tag, []))]
for role in ('train', 'test'):
    for name in ('sam31', 'micro1s3', 'rules'):
        S = {'in': [0, 0, 0], 'out': [0, 0, 0]}; n = 0
        for tag, v in b.items():
            if v['role'] != role or 'micro1s3' not in v or 'error' in v['micro1s3'] or 'error' in v['sam31']:
                continue
            n += 1
            ev = R.run_tag(d['tags'][tag], c, tag_name=tag)[0] if name == 'rules' else v[name]['events']
            m = R.match(drop_gray(tag, ev), drop_gray(tag, truth[tag]))
            for k in S:
                for i in range(3): S[k][i] += m[k][i]
        f = sum(2 * S[k][0] / max(1, S[k][1] + S[k][2]) for k in S) / 2
        print(role, name, n, 'F1 %.3f' % f, S)
