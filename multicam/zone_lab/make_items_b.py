"""moments where the LIVE logic (owner model + line; SAM teacher tracks and micro1s3) disagrees with the corrected truth,
not already reviewed -> appended to items.json as b### (no system side labels: those are the zone rules' sides)"""
import json, rules as R
exec(open('rescore.py').read().split('def drop_gray')[0])
items = json.load(open('items.json'))
items = [i for i in items if not i['id'].startswith('b')]
seen = [(i['tag'], i['t']) for i in items]
K = {'in': 'вход', 'out': 'выход'}
new = []
for tag, v in b.items():
    if 'micro1s3' not in v or 'error' in v['micro1s3'] or 'error' in v['sam31']:
        continue
    T = truth[tag]
    for name in ('sam31', 'micro1s3'):
        ev = v[name]['events']
        cand = [('missed', x) for x in T if not any(abs(e['t'] - x['t']) <= 6 and e['kind'] == x['kind'] for e in ev)] + \
               [('extra', e) for e in ev if not any(abs(e['t'] - x['t']) <= 6 and e['kind'] == x['kind'] for x in T)]
        for typ, e in cand:
            if any(tg == tag and abs(t - e['t']) <= 6 for tg, t in seen + [(n['tag'], n['t']) for n in new]):
                continue
            if any(abs(g - e['t']) <= 6 for g in gray.get(tag, [])):
                continue
            q = ('Был %s? (у тебя отмечен, боевая логика не увидела)' if typ == 'missed' else 'Был %s? (боевая логика увидела, у тебя не отмечен)') % K[e['kind']]
            new.append(dict(tag=tag, t=float(e['t']), kind=typ + '_' + e['kind'], q=q, note='боевая логика (%s)' % name,
                            pieces=sorted({r[R.PIECE] for r in d['tags'][tag]['rows'] if typ == 'missed' and r[R.ROOT] == str(e.get('person')) and abs(r[R.K] / 12.5 - e['t']) <= 5.2}), nosys=True, day=tag[5:13], src='live'))
for i, it in enumerate(new):
    it['id'] = 'b%03d' % i
json.dump(items + new, open('items.json', 'w'), ensure_ascii=False, indent=1)
print(len(new), [(n['tag'][-11:], n['kind'], round(n['t'], 1), n['note']) for n in new])
