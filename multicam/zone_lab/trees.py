"""13.10.2026: per-observation features (zones, legs, height drop, causal shop/door memory, pixel-kNN prior) ->
trees (depth 3/5/7) and a small random forest, trained leave-one-day-out on 'train' stretches; their p_in is the vote
inside the same event machinery as the rules (rules.run_tag with clf=...)."""
import gzip, json
from collections import defaultdict
import numpy as np
import rules as R

NAMES = ['f_band', 'f_shop', 'f_door', 'f_gal', 'shop_px', 'foot_shop', 'foot_door', 'foot_band', 'foot_gal', 'h',
         'h_ratio', 'legs', 'seen_shop', 'door_after_shop', 'pix_mean', 'pix_max', 'pix_frac', 'pix_low']


def features(data, pix):
    """{(tag, k, piece): vector}"""
    out = {}
    for tag, d in data['tags'].items():
        P = pix.get(tag, {})
        by = defaultdict(list)
        for r in d['rows']:
            by[r[R.PIECE]].append(r)
        for piece, xs in by.items():
            xs.sort(key=lambda r: r[R.K])
            seen, after, hs = 0, 0, []
            for r in xs:
                n = max(1, r[R.N]); h = r[R.Y2] - r[R.Y1]
                med = np.median(hs[-20:]) if hs else h
                v = [r[R.BAND] / n, r[R.SHOP] / n, r[R.DOOR] / n, r[R.GAL] / n, r[R.SHOP],
                     *(float(r[R.FOOT] == z) for z in ('shop', 'door', 'band', 'gallery')), h, h / max(1, med), r[R.LEGS],
                     seen, after] + (P.get('%d:%s' % (r[R.K], piece)) or [.5, .5, .5, .5])
                out[(tag, r[R.K], piece)] = v
                if r[R.SHOP] >= 1:
                    seen, after = 1, 0
                if r[R.DOOR] >= 20 and seen:
                    after = 1
                if r[R.LEGS] < .2:
                    hs.append(h)
    return out


def fit_lodo(data, F, make):
    """day -> model trained on the 'train' stretches of the other days"""
    days = sorted({t[5:13] for t in data['tags']})
    models = {}
    for day in days:
        X, y = [], []
        for tag, d in data['tags'].items():
            if d['role'] != 'train' or tag[5:13] == day:
                continue
            for r in d['rows']:
                if r[R.TRU] in ('in', 'out'):
                    X.append(F[(tag, r[R.K], r[R.PIECE])]); y.append(r[R.TRU] == 'in')
        models[day] = make().fit(np.array(X), np.array(y))
    return models
