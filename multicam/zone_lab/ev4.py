import eval3 as E, lab2 as L, lab4 as Q, itertools, sys
rows = {t: L.rows_of(E.tracks(t, 25, .4)) for t in E.tags}
def sc(c, roles=('train',), detail=False):
    S = {'in': [0, 0, 0], 'out': [0, 0, 0]}
    for t in E.tags:
        if E.d['tags'][t]['role'] not in roles:
            continue
        ev, _ = Q.events(rows[t], c); m = E.R.match(E.drop_gray(t, ev), E.drop_gray(t, E.truth[t]))
        if detail and (m['in'][0] < max(m['in'][1:]) or m['out'][0] < max(m['out'][1:])):
            print('  ', t, m)
        for k in S:
            for i in range(3): S[k][i] += m[k][i]
    return round(sum(2 * S[k][0] / max(1, S[k][1] + S[k][2]) for k in S) / 2, 3), S
