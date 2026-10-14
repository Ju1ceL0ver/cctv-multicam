"""15.10 owner: is the SAM 3.1 teacher better than the live door where the door fails? Events of the teacher
(zone_baseline.json 'sam31') and of the live setting vs the owner's /doorside truth, overall and at the live errors."""
import json, os
exec(open('rescore.py').read().split('def drop_gray')[0])
def dg(tag, ev): return [e for e in ev if not any(abs(e['t'] - g) <= 6 for g in gray.get(tag, []))]
errs = json.load(open(os.path.join('..', 'data', 'door_v2', 'err_live', 'index.json')))
for name, key in (('teacher SAM 3.1', 'sam31'), ('old live SAM micro', 'micro1s3')):
    for role in ('train', 'test'):
        r_ = s_ = t_ = 0
        for t, v in d['tags'].items():
            if v['role'] != role or key not in b.get(t, {}): continue
            m = R.match(dg(t, b[t][key]['events']), dg(t, truth[t]))
            for k in ('in', 'out'): r_ += m[k][0]; s_ += m[k][1]; t_ += m[k][2]
        print('%-20s %-5s F1 %.3f (right %d, said %d, truth %d)' % (name, role, 2 * r_ / max(1, s_ + t_), r_, s_, t_))
print('at the live door errors (+-6 s window, same kind):')
ok = 0
for e in errs:
    t = e['tag']; ev = dg(t, b[t]['sam31']['events']); tr = dg(t, truth[t])
    lo, hi = e['t'] - 6, e['t'] + 6
    a = sum(1 for x in ev if x['kind'] == e['kind'] and lo <= x['t'] <= hi)
    c = sum(1 for x in tr if x['kind'] == e['kind'] and lo <= x['t'] <= hi)
    good = a == c
    ok += good
    print('  %s %-6s %-3s %s %6.1f  truth %d  teacher %d  %s' % (e['n'], e['what'], e['kind'], t, e['t'], c, a, 'teacher right' if good else 'teacher wrong'))
print('teacher right at %d of %d live errors' % (ok, len(errs)))
