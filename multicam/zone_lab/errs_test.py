import glob, json
exec(open('rescore.py').read().split('def drop_gray')[0])
def dg(tag, ev): return [e for e in ev if not any(abs(e['t'] - g) <= 6 for g in gray.get(tag, []))]
b = json.load(open('zone_baseline.json'))
O = {}
for f in glob.glob('big2_events_*.json'):
    O.update(json.load(open(f)))
tags = ['door_20260917_08691', 'door_20260918_04917', 'door_20260918_06803', 'door_20260918_13748', 'door_20260918_15209', 'door_20260918_15826',
        'door_20260919_00095', 'door_20260919_23237', 'door_20260919_24209', 'door_20260919_24505', 'door_20260919_27663']
SYS = {'live_micro1s3': lambda t: b[t]['micro1s3']['events'], 'sam_teacher': lambda t: b[t]['sam31']['events'],
       'dsc_rules_cont25': lambda t: O['25_0.25_rules'][t], 'dsc_geo': lambda t: O['25_0.4_lab4'][t], 'dsc_rules': lambda t: O['25_0.4_rules'][t]}
for t in tags:
    tr = dg(t, truth[t])
    print('==', t, 'truth:', [(e['kind'], round(e['t'], 1)) for e in sorted(tr, key=lambda e: e['t'])])
    for name, get in SYS.items():
        ev = dg(t, get(t))
        miss = [(x['kind'], round(x['t'], 1)) for x in tr if not any(abs(e['t'] - x['t']) <= 6 and e['kind'] == x['kind'] for e in ev)]
        extra = [(e['kind'], round(e['t'], 1)) for e in ev if not any(abs(e['t'] - x['t']) <= 6 and e['kind'] == x['kind'] for x in tr)]
        if miss or extra:
            print('   %-17s missed %s  extra %s' % (name, miss, extra))
print('\ncounts in/out per stretch (truth | live | dsc_rules_cont25 | dsc_geo):')
for t in tags:
    tr = dg(t, truth[t])
    c = lambda ev: (sum(e['kind'] == 'in' for e in ev), sum(e['kind'] == 'out' for e in ev))
    row = [c(tr)] + [c(dg(t, SYS[n](t))) for n in ('live_micro1s3', 'dsc_rules_cont25', 'dsc_geo')]
    if len(set(row)) > 1:
        print(t, row, 'pred times dsc_rules:', [(e['kind'], round(e['t'], 1)) for e in sorted(dg(t, SYS['dsc_rules_cont25'](t)), key=lambda e: e['t'])])
