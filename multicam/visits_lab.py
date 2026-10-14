"""Trajectory lab on the owner-labelled clips (gt_identity_*.json): smoothing alone, association aside.

Per labelled person: floor points from both cameras (foot, else head at own height). Measures
  * roughness raw vs smoothed (p90 speed, share of steps faster than 2.5 m/s);
  * accuracy by the other camera: smooth on one camera, compare with the other camera's
    foot-visible points at the same moments (median / p90 metres). Raw = nearest own point.
usage: visits_lab.py [clip ...]"""
import sys, os, json, numpy as np
ROOT = os.path.dirname(os.path.abspath(__file__)); os.chdir(ROOT); sys.path.insert(0, ROOT)
from person3d import Camera, place, map_cam1
from topview import UNIT
from fusion2 import REG
from clipdata import load
import visit_traj as vt

CLIPS = ['c100054', 'c101054', 'c103700', 'c112000', 'c155236', 'c171400', 'c183400']


def people_points(cams, dets, P, gt):
    out = {}
    for cam in ('cam1', 'cam2'):
        for k, g in gt[cam].items():
            if not g.startswith('P'):
                continue
            out.setdefault(g, {}).setdefault(cam, []).append(int(k))
    res = {}
    for g, per in out.items():
        rows = []
        for cam, ii in per.items():
            ii = np.array(sorted(ii))
            d = dets[cam][ii]; pl = P[cam]
            vis = pl['vis'][ii]; xy = pl['xy'][ii].copy(); h = pl['h'][ii]
            hs = h[vis & np.isfinite(h) & (h > 1.05) & (h < 2.15)]
            if len(hs) >= 5:            # head-only frames placed at this person's own height
                xyh = cams[cam].on_plane(d[:, 8:10], z=float(np.median(hs)) / UNIT) * UNIT
                if cam == 'cam1':
                    xyh = map_cam1(xyh, REG)
                xy = np.where(vis[:, None], xy, xyh)
            rows.append((cam, d[:, 0], xy, vis))
        res[g] = rows
    return res


def run(clip):
    dets, feats, meta, embs = load(clip)
    gt = json.load(open(os.path.join('data', 'raw_clips', clip, 'gt_identity_yolo26x-seg.json')))
    calib = json.load(open('data/calib_final.json'))
    cams = {c: Camera(c, calib) for c in ('cam1', 'cam2')}
    P = place(cams, REG, dets)
    stats = []
    for g, rows in people_points(cams, dets, P, gt).items():
        t = np.concatenate([r[1] for r in rows]); xy = np.concatenate([r[2] for r in rows])
        foot = np.concatenate([r[3] for r in rows])
        if len(t) < 20 or t.max() - t.min() < 5:
            continue
        o = np.argsort(t, kind='stable')
        raw = vt.roughness(t[o], xy[o])
        grid, sxy, sp, keep = vt.smooth(t, xy, foot)
        row = {'clip': clip, 'person': g, 'seconds': round(float(t.max() - t.min()), 1), 'n': int(len(t)),
               'cams': sorted({r[0] for r in rows}), 'foot_share': round(float(foot.mean()), 2),
               'raw': raw, 'smooth': vt.roughness(grid, sxy), 'rejected': int((~keep).sum())}
        by = {r[0]: r for r in rows}
        if len(by) == 2:                 # other-camera check
            errs_s, errs_r = [], []
            for a, b in (('cam1', 'cam2'), ('cam2', 'cam1')):
                _, ta, xa, fa = by[a]; _, tb, xb, fb = by[b]
                if len(ta) < 10:
                    continue
                ga, sa, _, _ = vt.smooth(ta, xa, fa)
                sel = fb & (tb >= ga[0]) & (tb <= ga[-1])
                if not sel.any():
                    continue
                oa = np.argsort(ta)
                j = np.clip(np.searchsorted(ta[oa], tb[sel]), 0, len(ta) - 1)
                near = np.abs(ta[oa][j] - tb[sel]) < 0.5      # same moments for both: camera a saw the person
                tq, xq = tb[sel][near], xb[sel][near]
                px = np.c_[np.interp(tq, ga, sa[:, 0]), np.interp(tq, ga, sa[:, 1])]
                errs_s.append(np.linalg.norm(px - xq, axis=1))
                errs_r.append(np.linalg.norm(xa[oa][j][near] - xq, axis=1))
            if errs_s:
                es = np.concatenate(errs_s); er = np.concatenate(errs_r)
                row['xcam_smooth'] = [round(float(np.median(es)), 2), round(float(np.percentile(es, 90)), 2), int(len(es))]
                if len(er):
                    row['xcam_raw'] = [round(float(np.median(er)), 2), round(float(np.percentile(er, 90)), 2), int(len(er))]
        stats.append(row)
    return stats


if __name__ == '__main__':
    clips = sys.argv[1:] or CLIPS
    allrows = []
    for c in clips:
        try:
            allrows += run(c)
        except Exception as e:
            print(c, 'failed', repr(e))
    os.makedirs('data/visits_lab', exist_ok=True)
    json.dump(allrows, open('data/visits_lab/traj_stats.json', 'w'), indent=1)

    def agg(key, sub, f=np.median):
        v = [r[key][sub] for r in allrows if r.get(key) and sub in r[key]]
        return round(float(f(v)), 3) if v else None
    print('people', len(allrows))
    for k in ('raw', 'smooth'):
        print(k, 'v_p90 median', agg(k, 'v_p90'), 'over_vmax mean', agg(k, 'over_vmax', np.mean))
    for k in ('xcam_raw', 'xcam_smooth'):
        v = [r[k] for r in allrows if k in r]
        if v:
            w = np.array([x[2] for x in v]); m = np.array([x[0] for x in v]); p = np.array([x[1] for x in v])
            print(k, 'people %d points %d  median %.2f m  p90 %.2f m (weighted)' % (len(v), w.sum(), np.average(m, weights=w), np.average(p, weights=w)))
