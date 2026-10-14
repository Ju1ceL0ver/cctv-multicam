import json, rules as R
d = R.load('zone_feats.json.gz')
c = dict(R.BASE, shop=1, first=0, occl_legs=.6, occl_h=.6, conf=1, pair_s=3)
K = {'in': 'вход', 'out': 'выход'}
items = []
for tag, t in sorted(d['tags'].items()):
    ev, so = R.run_tag(t, c, tag_name=tag)
    person = dict(R.LAST_PERSON)
    tr = t['events']
    near = lambda pred, tt: sorted({r[R.PIECE] for r in t['rows'] if pred(r) and abs(r[R.K] / 12.5 - tt) <= 5.2})
    done = set()
    for x in tr:
        pair = [f for f in tr if f is not x and f['person'] == x['person'] and f['kind'] != x['kind'] and abs(f['t'] - x['t']) < .5]
        if pair:
            key = (x['person'], round(min(x['t'], pair[0]['t']), 1))
            if key in done:
                continue
            done.add(key)
            items.append(dict(tag=tag, t=x['t'], kind='pair', q='Человек вышел и тут же вошёл (у тебя обе отметки в одну долю секунды). Это правда было?',
                              note='твой №%s' % x['person'], pieces=near(lambda r: r[R.ROOT] == str(x['person']), x['t'])))
            continue
        if not any(abs(e['t'] - x['t']) <= 6 and e['kind'] == x['kind'] for e in ev):
            items.append(dict(tag=tag, t=x['t'], kind='missed_' + x['kind'], q='Был %s? (у тебя отмечен, система не увидела)' % K[x['kind']],
                              note='твой №%s' % x['person'], pieces=near(lambda r: r[R.ROOT] == str(x['person']), x['t'])))
    for e in ev:
        if not any(abs(e['t'] - x['t']) <= 6 and e['kind'] == x['kind'] for x in tr):
            items.append(dict(tag=tag, t=e['t'], kind='extra_' + e['kind'], q='Был %s? (система увидела, у тебя не отмечен)' % K[e['kind']],
                              note='трек системы %s' % e['w'], pieces=near(lambda r: person.get(r[R.PIECE]) == e['w'], e['t'])))
    for it in items:
        if it['tag'] != tag or 'sides' in it:
            continue
        it['sides'] = [[r[R.K], r[R.PIECE], so.get((r[R.PIECE], r[R.K])), r[R.TRU]] for r in t['rows']
                       if abs(r[R.K] / 12.5 - it['t']) <= 5.4]
for i, it in enumerate(items):
    it['id'] = 'z%03d' % i
    it['day'] = it['tag'][5:13]
json.dump(items, open('items.json', 'w'), ensure_ascii=False, indent=1)
print(len(items), [(i['tag'][-11:], i['kind'], round(i['t'], 1), len(i['pieces'])) for i in items])
