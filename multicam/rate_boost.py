"""Draft quality 1..5 from a few dozen plain numbers, gradient boosting (26.09).

Two light CNNs over the same inputs (train_rate.py) learned the 333 training frames by heart and
did no better than this on the held-out ones, so the scorer is boosting over hand-made features
until there are many more scores. The features come from the draft and the same background
differences (rate_features.py + the prebuilt file backgrounds), all cheap in real time:

  per background map (5-min, 30-min, other days, file):  share of the frame that differs; that
  differs where no person is drawn (a missed person); drawn where nothing differs (a false one);
  their overlap (IoU); blobs of difference with no person on them, how many and the largest;
  mean difference inside the drawn people; the 99th percentile outside them;
  the draft itself: people, share of the frame, tiny pieces, the smallest piece, camera.

Measured: a stratified 80/20 split and 5-fold, both by recording file (a file's frames never on both sides), MSE against 'always the
mean', Spearman, AUC of 4-5 vs 1-3, and what a >= 4 cut keeps. Score 0 is left out.
Model and report: data/rate/boost/{model.joblib, report.json, log.txt}.

  python rate_boost.py select [--keep 4.72] [--crowd 3]
      score every draft that has its background maps (rate_features.py --all) and decide:
      fewer than --crowd people: keep if the score >= --keep, else drop; --crowd or more people:
      'owner' -- the scorer is weakest exactly there and these frames are the most valuable, so
      the owner looks at them (lowest scores first) instead of the scorer throwing them away.
      Thresholds of 26.09, picked on the training part only (<= 5% of bad kept there): on the
      held-out 90 frames 1 bad of 22 passed and 28% of the good 0-2-people frames were kept.
      -> data/rate/selection.json

  python rate_boost.py classify
      good (4-5) vs bad (1-3) as classification, side by side with the regression on the same
      split and threshold rule -> data/rate/boost/{classify.json, model_cls.joblib}

  python rate_boost.py [--seed 0]
      a stratified 80/20 split by file (the main number), 5-fold by file as a check, then the
      model is fit on everything and saved
"""
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
MAPS = ('short', 'long', 'days', 'file')
THR = (12, 12, 4, 12)          # 'differs' per map: Lab distance for three, spreads for the days map
PER_MAP = ('fg', 'fg_nomask', 'mask_nofg', 'iou', 'miss_blobs', 'miss_max', 'd_in_mask', 'p99_out')
NAMES = ['people', 'mask_share', 'tiny_pieces', 'min_piece', 'cam2'] + ['%s_%s' % (m, f) for m in MAPS for f in PER_MAP]
# v2 (26.09): the worst person of the draft, small missed people, and what our student sees
SHAPE = ['max_parts', 'min_solidity', 'max_width_to_height', 'min_fill_of_box', 'min_fg_support', 'max_height_share']
SMALL = ['small_miss_blobs', 'small_miss_tall', 'small_miss_max_height']
STUD = ['st_people', 'st_people_low', 'st_unmatched', 'st_unmatched_maxconf', 'draft_unmatched', 'st_draft_iou', 'min_match_iou']
NAMES_V2 = NAMES + SHAPE + SMALL + STUD
STUDENT = os.path.join(HERE, 'runs', 'student_seg_all_n', 'weights', 'best.pt')
_student = {}


def student_view(ident, image):
    """Our student yolo26n-seg on the frame at a low threshold, cached: people as a label map
    640x360 (1..n, in order of confidence) and their confidences. -> (labels, confs)"""
    import cv2
    folder = os.path.join(HERE, 'data', 'rate', 'student')
    png, js = os.path.join(folder, ident + '.png'), os.path.join(folder, ident + '.json')
    if os.path.exists(png) and os.path.exists(js):
        return cv2.imread(png, cv2.IMREAD_UNCHANGED), json.load(open(js))
    if 'm' not in _student:
        from ultralytics import YOLO
        _student['m'] = YOLO(STUDENT, task='segment')
    os.makedirs(folder, exist_ok=True)
    img = cv2.imread(str(image))
    res = _student['m'].predict(img, imgsz=1088, conf=0.10, classes=[0], retina_masks=True, verbose=False)[0]
    lab = np.zeros((360, 640), np.uint8)
    confs = []
    if res.masks is not None:
        ms = res.masks.data.cpu().numpy()
        cs = res.boxes.conf.cpu().numpy()
        for k in np.argsort(-cs)[:250]:                  # most confident drawn last = on top
            confs.append(float(cs[k]))
        for n, k in reversed(list(enumerate(np.argsort(-cs)[:250], 1))):
            lab[cv2.resize(ms[k].astype(np.uint8), (640, 360), interpolation=cv2.INTER_NEAREST) > 0] = n
    cv2.imwrite(png, lab)
    json.dump(confs, open(js, 'w'))
    return lab, confs


def features_v2(ident, image, draft, feat):
    """features() plus the worst person of the draft, small missed people and the student."""
    import cv2
    import train_rate
    f = features(ident, image, draft, feat)
    u8, d = train_rate.load_item(image, draft, feat, ident)
    d = d.astype(np.float32)
    lab = cv2.imread(str(draft), cv2.IMREAD_UNCHANGED)
    lab = cv2.resize(lab[:, :, 0] if lab.ndim == 3 else lab, (train_rate.W, train_rate.H), interpolation=cv2.INTER_NEAREST)
    m = lab > 0
    fg = np.max(d[[0, 1, 3]], axis=0) > 12
    people = [v for v in np.unique(lab) if v]
    parts, solid, wh, fill, supp, hs = [], [], [], [], [], []
    for v in people:
        pm = (lab == v).astype(np.uint8)
        n, cc, st, _ = cv2.connectedComponentsWithStats(pm)
        parts.append(int((st[1:, 4] >= 15).sum()))
        cs, _ = cv2.findContours(pm, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        hull = cv2.convexHull(np.vstack(cs)) if cs else None
        solid.append(pm.sum() / max(1.0, cv2.contourArea(hull)) if hull is not None else 1.0)
        ys, xs = np.nonzero(pm)
        w, h = xs.max() - xs.min() + 1, ys.max() - ys.min() + 1
        wh.append(w / h)
        fill.append(pm.sum() / float(w * h))
        supp.append(fg[pm > 0].mean())
        hs.append(h / 360.0)
    f += [max(parts, default=0), min(solid, default=1), max(wh, default=0), min(fill, default=1),
          min(supp, default=1), max(hs, default=0)]
    # small missed people: blobs of difference with nobody drawn near, down to 60 px, tall ones apart
    near = cv2.dilate(m.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
    blob = cv2.morphologyEx((fg & ~near).astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, cc, st, _ = cv2.connectedComponentsWithStats(blob)
    small = [s for s in st[1:] if s[4] >= 60]
    tall = [s for s in small if s[3] >= 1.3 * s[2]]
    f += [len(small), len(tall), max((s[3] for s in tall), default=0) / 360.0]
    # the student against the draft
    sl, sc = student_view(ident, image)
    sids = [k for k in range(1, len(sc) + 1)]
    hi = [k for k in sids if sc[k - 1] >= 0.25]
    lo = [k for k in sids if 0.10 <= sc[k - 1] < 0.25]

    def best_iou(a, other, ids):
        best = 0.0
        for k in ids:
            b = other == k
            inter = (a & b).sum()
            if inter:
                best = max(best, inter / float((a | b).sum()))
        return best
    st_un = [k for k in hi if best_iou(sl == k, lab, people) < 0.3]
    dr_match = [best_iou(lab == v, sl, hi) for v in people]
    sm = np.isin(sl, hi)
    f += [len(hi), len(lo), len(st_un), max((sc[k - 1] for k in st_un), default=0.0),
          sum(1 for x in dr_match if x < 0.3), (sm & m).sum() / max(1, (sm | m).sum()), min(dr_match, default=1.0)]
    return f



def features(ident, image, draft, feat):
    import cv2
    import train_rate
    u8, d = train_rate.load_item(image, draft, feat, ident)
    m = u8[:, :, 3] > 0
    d = d.astype(np.float32)
    lab = cv2.imread(str(draft), cv2.IMREAD_UNCHANGED)
    lab = cv2.resize(lab[:, :, 0] if lab.ndim == 3 else lab, (train_rate.W, train_rate.H), interpolation=cv2.INTER_NEAREST)
    areas = [int((lab == v).sum()) for v in np.unique(lab) if v]
    f = [len(areas), m.mean(), sum(a < 150 for a in areas), min(areas) if areas else 0, ident.split('_')[1] == 'cam2']
    for k, thr in enumerate(THR):
        fg = cv2.morphologyEx((d[k] > thr).astype(np.uint8), cv2.MORPH_OPEN, np.ones((5, 5), np.uint8)) > 0
        n, cc, st, _ = cv2.connectedComponentsWithStats(fg.astype(np.uint8))
        miss = [s[4] for j, s in enumerate(st[1:], 1) if s[4] > 300 and m[cc == j].mean() < 0.2]
        f += [fg.mean(), (fg & ~m).mean(), (m & ~fg).mean(), (fg & m).sum() / max(1, (fg | m).sum()), len(miss),
              max(miss) if miss else 0, d[k][m].mean() if m.any() else 0, np.percentile(d[k][~m], 99)]
    return f


def make_classifier():
    """good (4-5) against bad (1-3), probability of good; the same trees as the regression."""
    from sklearn.ensemble import GradientBoostingClassifier
    return GradientBoostingClassifier(n_estimators=150, max_depth=2, learning_rate=0.05, subsample=0.8, random_state=0)


def pick_threshold(p, bad, target):
    """The loosest cut on p that keeps at most `target` of the bad frames (on the training part)."""
    for t in np.unique(np.concatenate([p, [np.inf]])):
        if ((p >= t) & bad).sum() <= target * bad.sum():
            return float(t)
    return float('inf')


def classify(seed=0, crowd=3):
    """Classification against the regression, the same split and the same threshold rule:
    thresholds picked on out-of-fold predictions inside the training part, per crowd group, and
    applied once to the test part."""
    import joblib
    import rate
    from scipy.stats import spearmanr
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import GroupKFold, StratifiedGroupKFold, cross_val_predict
    import train_rate
    out = os.path.join(HERE, 'data', 'rate', 'boost')
    logf = open(os.path.join(out, 'log.txt'), 'a', encoding='utf-8')

    def log(s):
        print(s, flush=True)
        logf.write(time.strftime('%Y-%m-%d %H:%M:%S ') + s + '\n')
        logf.flush()
    items, r = rate.index(HERE, force=True), rate.ratings(HERE)
    fdir = os.path.join(HERE, 'data', 'rate', 'features')
    ids = sorted(i for i, v in r.items() if 1 <= v['score'] <= 5 and i in items and os.path.exists(os.path.join(fdir, i + '.npz')))
    X = np.array([features_v2(i, items[i][0], items[i][1], os.path.join(fdir, i + '.npz')) for i in ids], float)
    y = np.array([r[i]['score'] for i in ids], float)
    g = np.array([train_rate.group_of(i) for i in ids])
    good, many = y >= 4, X[:, 0] >= crowd
    tr, te = next(StratifiedGroupKFold(5, shuffle=True, random_state=seed).split(X, y, g))
    log('=' * 80)
    log('classification vs regression: %d frames (good %d, bad %d), train %d / test %d (test bad %d, of them crowded %d)'
        % (len(y), good.sum(), (~good).sum(), len(tr), len(te), (~good[te]).sum(), (~good[te] & many[te]).sum()))
    res = {}
    for name in ('regression', 'classification'):
        if name == 'regression':
            mk, oof_kw = make_model, {}
            fit = lambda Xa, ya: make_model().fit(Xa, ya)
            pred = lambda m, Xa: m.predict(Xa)
            target_y = y
        else:
            mk, oof_kw = make_classifier, {'method': 'predict_proba'}
            fit = lambda Xa, ya: make_classifier().fit(Xa, ya >= 4)
            pred = lambda m, Xa: m.predict_proba(Xa)[:, 1]
            target_y = good
        ptr = cross_val_predict(mk(), X[tr], target_y[tr], groups=g[tr], cv=GroupKFold(5), **oof_kw)
        ptr = ptr[:, 1] if ptr.ndim == 2 else ptr
        pte = pred(fit(X[tr], y[tr]), X[te])
        auc = roc_auc_score(good[te], pte)
        cv_all = cross_val_predict(mk(), X, target_y, groups=g, cv=GroupKFold(5), **oof_kw)
        cv_all = cv_all[:, 1] if cv_all.ndim == 2 else cv_all
        log('%s: test AUC good vs bad %.3f, Spearman with the 1-5 score %.3f; 5-fold by file AUC %.3f'
            % (name, auc, spearmanr(pte, y[te])[0], roc_auc_score(good, cv_all)))
        res[name] = {'test_auc': auc, 'cv_auc': roc_auc_score(good, cv_all), 'cuts': {}}
        for target in (0.0, 0.05, 0.10, 0.20):
            t_few = pick_threshold(ptr[~many[tr]], ~good[tr][~many[tr]], target)
            t_many = pick_threshold(ptr[many[tr]], ~good[tr][many[tr]], target)
            k = pte >= np.where(many[te], t_many, t_few)
            row = {}
            for grp, m in (('all', np.ones(len(te), bool)), ('0-2', ~many[te]), ('3+', many[te])):
                row[grp] = [int((k & ~good[te] & m).sum()), int((~good[te] & m).sum()), int((k & good[te] & m).sum()), int((good[te] & m).sum())]
            res[name]['cuts'][target] = {'thr_0_2': t_few, 'thr_3plus': t_many, 'test': row}
            log('  bad <= %2d%% on train: cut 0-2 people %.3f, 3+ people %.3f | test bad kept %d/%d, good kept %d/%d | 0-2: bad %d/%d good %d/%d | 3+: bad %d/%d good %d/%d'
                % (100 * target, t_few, t_many, *row['all'], *row['0-2'], *row['3+']))
    clf = make_classifier().fit(X, good)
    joblib.dump({'model': clf, 'names': NAMES_V2, 'thr': THR, 'trained_on': len(ids), 'kind': 'P(good), good = 4-5'},
                os.path.join(out, 'model_cls.joblib'))
    json.dump(res, open(os.path.join(out, 'classify.json'), 'w'), indent=1)
    log('saved %s' % os.path.join(out, 'model_cls.joblib'))


def make_model():
    from sklearn.ensemble import GradientBoostingRegressor
    return GradientBoostingRegressor(n_estimators=150, max_depth=2, learning_rate=0.05, subsample=0.8, random_state=0)


def summary(y, p):
    from scipy.stats import spearmanr
    from sklearn.metrics import roc_auc_score
    p = np.clip(p, 1, 5)
    good = y >= 4
    out = {'n': int(len(y)), 'mse': float(((p - y) ** 2).mean()), 'mse_mean_only': float(y.var()),
           'mae': float(np.abs(p - y).mean()), 'spearman': float(spearmanr(p, y)[0]), 'auc_good': float(roc_auc_score(good, p)),
           'mean_pred_by_score': {int(s): round(float(p[y == s].mean()), 2) for s in sorted(set(y.tolist()))}}
    for thr in (3.5, 4.0, 4.3):
        keep = p >= thr
        out['keep>=%.1f' % thr] = {'kept': int(keep.sum()), 'good_share': round(float(good[keep].mean()), 3) if keep.any() else None,
                                   'good_kept': round(float(keep[good].mean()), 3), 'bad_kept': int((keep & ~good).sum()),
                                   'bad_total': int((~good).sum())}
    return out


def main():
    import argparse
    import joblib
    import rate
    from sklearn.model_selection import GroupKFold, cross_val_predict
    import train_rate
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()
    out = os.path.join(HERE, 'data', 'rate', 'boost')
    os.makedirs(out, exist_ok=True)
    logf = open(os.path.join(out, 'log.txt'), 'a', encoding='utf-8')

    def log(s):
        print(s, flush=True)
        logf.write(time.strftime('%Y-%m-%d %H:%M:%S ') + s + '\n')
        logf.flush()
    items, r = rate.index(HERE, force=True), rate.ratings(HERE)
    fdir = os.path.join(HERE, 'data', 'rate', 'features')
    ids = sorted(i for i, v in r.items() if 1 <= v['score'] <= 5 and i in items and os.path.exists(os.path.join(fdir, i + '.npz')))
    skipped = sum(1 for i, v in r.items() if 1 <= v['score'] <= 5 and i not in ids)
    t0 = time.time()
    X = np.array([features_v2(i, items[i][0], items[i][1], os.path.join(fdir, i + '.npz')) for i in ids], float)
    y = np.array([r[i]['score'] for i in ids], float)
    groups = [train_rate.group_of(i) for i in ids]
    log('=' * 80)
    log('%d scored frames 1..5 (%d without background maps left out), %d files, scores %s; features %.0f s'
        % (len(ids), skipped, len(set(groups)), {s: int((y == s).sum()) for s in range(1, 6)}, time.time() - t0))
    from sklearn.metrics import roc_auc_score
    for name, cols in (('old features (%d)' % len(NAMES), slice(0, len(NAMES))), ('new features (%d)' % len(NAMES_V2), slice(None))):
        aucs = [roc_auc_score(y >= 4, np.clip(cross_val_predict(make_model(), X[:, cols], y, groups=groups,
                              cv=GroupKFold(5, shuffle=True, random_state=s)), 1, 5)) for s in range(5)]
        log('%s: 5-fold by file AUC 4-5 vs 1-3 over 5 shuffles %.3f +- %.3f' % (name, np.mean(aucs), np.std(aucs)))
    p = cross_val_predict(make_model(), X, y, groups=groups, cv=GroupKFold(5))
    cv = summary(y, p)
    log('5-fold by file: MSE %.3f (always the mean %.3f), MAE %.3f, Spearman %.3f, AUC 4-5 vs 1-3 %.3f'
        % (cv['mse'], cv['mse_mean_only'], cv['mae'], cv['spearman'], cv['auc_good']))
    log('  mean prediction by score: %s' % cv['mean_pred_by_score'])
    for k in ('keep>=3.5', 'keep>=4.0', 'keep>=4.3'):
        v = cv[k]
        log('  %s: keeps %d of %d, good among kept %.0f%% (all frames %.0f%%), keeps %.0f%% of the good, bad kept %d of %d'
            % (k, v['kept'], len(y), 100 * (v['good_share'] or 0), 100 * (y >= 4).mean(), 100 * v['good_kept'], v['bad_kept'], v['bad_total']))
    per_fold = []
    for k, (tr, te) in enumerate(GroupKFold(5).split(X, y, groups)):
        per_fold.append(round(summary(y[te], p[te])['auc_good'], 3))
    log('  AUC per fold: %s' % per_fold)
    # the main check: one stratified split, 20% test, score classes in the same shares on both sides,
    # whole recording files on one side only
    from sklearn.model_selection import StratifiedGroupKFold
    tr, te = next(StratifiedGroupKFold(5, shuffle=True, random_state=a.seed).split(X, y, groups))
    sm = make_model().fit(X[tr], y[tr])
    split = summary(y[te], sm.predict(X[te]))
    split['train_scores'] = {s: int((y[tr] == s).sum()) for s in range(1, 6)}
    split['test_ids'] = [ids[k] for k in te]
    base = float(((y[te] - y[tr].mean()) ** 2).mean())
    log('stratified split (seed %d): train %d frames %s, test %d frames %s, no file on both sides'
        % (a.seed, len(tr), split['train_scores'], len(te), {s: int((y[te] == s).sum()) for s in range(1, 6)}))
    log('  test: MSE %.3f (always the train mean %.3f), RMSE %.3f, MAE %.3f, Spearman %.3f, AUC 4-5 vs 1-3 %.3f'
        % (split['mse'], base, np.sqrt(split['mse']), split['mae'], split['spearman'], split['auc_good']))
    log('  mean prediction by score: %s' % split['mean_pred_by_score'])
    for k in ('keep>=3.5', 'keep>=4.0', 'keep>=4.3'):
        v = split[k]
        log('  %s: keeps %d of %d, good among kept %.0f%% (test %.0f%%), keeps %.0f%% of the good, bad kept %d of %d'
            % (k, v['kept'], len(te), 100 * (v['good_share'] or 0), 100 * (y[te] >= 4).mean(), 100 * v['good_kept'], v['bad_kept'], v['bad_total']))
    model = make_model().fit(X, y)
    imp = sorted(zip(model.feature_importances_, NAMES_V2), reverse=True)
    log('  top features: %s' % ', '.join('%s %.3f' % (n, v) for v, n in imp[:10]))
    joblib.dump({'model': model, 'names': NAMES_V2, 'thr': THR, 'trained_on': len(ids)}, os.path.join(out, 'model.joblib'))
    json.dump({'split': split, 'cv': cv, 'auc_per_fold': per_fold, 'importance': {n: float(v) for v, n in imp},
               'predictions_cv': {i: [int(a), round(float(b), 3)] for i, a, b in zip(ids, y, p)}},
              open(os.path.join(out, 'report.json'), 'w'), indent=1)
    log('saved %s' % os.path.join(out, 'model.joblib'))


def _one(job):
    import cv2
    cv2.setNumThreads(1)
    ident, image, draft, feat = job
    try:
        return ident, features_v2(ident, image, draft, feat)
    except Exception as exc:                          # a broken file must not stop 13 000 others
        return ident, repr(exc)


def select(keep=4.72, crowd=3, workers=12):
    import joblib
    from multiprocessing import Pool
    import rate
    items = rate.index(HERE, force=True)
    fdir = os.path.join(HERE, 'data', 'rate', 'features')
    m = joblib.load(os.path.join(HERE, 'data', 'rate', 'boost', 'model.joblib'))
    jobs = [(i, str(im), str(dr), os.path.join(fdir, i + '.npz')) for i, (im, dr) in sorted(items.items())
            if os.path.exists(os.path.join(fdir, i + '.npz'))]
    t0 = time.time()
    print('%d drafts, %d with background maps' % (len(items), len(jobs)), flush=True)
    with Pool(workers) as pool:
        res = pool.map(_one, jobs, chunksize=16)
    bad = {i: f for i, f in res if isinstance(f, str)}
    ok = [(i, f) for i, f in res if not isinstance(f, str)]
    X = np.array([f for _, f in ok], float)
    pred = np.clip(m['model'].predict(X), 1, 5)
    scores = rate.ratings(HERE)
    out = {}
    for (i, f), p in zip(ok, pred):
        people = int(f[0])
        decision = 'owner' if people >= crowd else ('keep' if p >= keep else 'drop')
        out[i] = {'pred': round(float(p), 3), 'people': people, 'decision': decision}
        if i in scores:
            out[i]['owner_score'] = scores[i]['score']
    counts = {d: sum(1 for v in out.values() if v['decision'] == d) for d in ('keep', 'drop', 'owner')}
    summary = {'at': time.strftime('%Y-%m-%dT%H:%M:%S'), 'keep': keep, 'crowd': crowd, 'model_trained_on': m['trained_on'],
               'drafts': len(items), 'scored': len(out), 'no_maps': len(items) - len(jobs), 'failed': bad, 'counts': counts,
               'seconds': round(time.time() - t0)}
    json.dump({'summary': summary, 'frames': out}, open(os.path.join(HERE, 'data', 'rate', 'selection.json'), 'w'), indent=0)
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'classify':
        classify()
    elif len(sys.argv) > 1 and sys.argv[1] == 'select':
        import argparse
        ap = argparse.ArgumentParser()
        ap.add_argument('cmd')
        ap.add_argument('--keep', type=float, default=4.72)
        ap.add_argument('--crowd', type=int, default=3)
        ap.add_argument('--workers', type=int, default=12)
        a = ap.parse_args()
        select(a.keep, a.crowd, a.workers)
    else:
        main()
