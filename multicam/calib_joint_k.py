"""Joint calibration with the lens distortion constrained by plumb lines.

Same data as calib_joint (hand-read tile crossings, floor seam ridges, vertical
edges, roll prior) plus straight edge chains that must come out straight. The
floor data alone sits near the image centre where distortion is weak, and the
'vertical' edges are mostly leaning display panels -- together they let k1
collapse to ~0 while straight walls at the image border clearly bend.

usage: python3 calib_joint_k.py OUT.json [fixed_k1]"""
import sys, cv2, json, numpy as np
from scipy.optimize import least_squares
import calib_final as cf
from calib_joint import load, cam_x, CAMS, W, H
from plumb import chains, straightness


def plumb_res(zc, chain_list):
    K, dist, rv, tv, rho = cf.unpack(zc)
    out = []
    for c in chain_list:
        u = cv2.undistortPoints(c.reshape(-1, 1, 2), K, dist, P=K).reshape(-1, 2)
        m = u.mean(0)
        vt = np.linalg.svd(u - m)[2]
        out.append(((u - m) @ vt[1]) / np.sqrt(len(u)) * 3.0)
    return np.concatenate(out) if out else np.zeros(0)


def plumb_chains(cam):
    img = cv2.imread('data/%s_bg.jpg' % cam)
    ks = np.round(np.arange(-0.45, 0.151, 0.025), 3)
    keep = []
    for c in chains(img):
        rms = np.array([straightness(c, k)[0] for k in ks])
        j = int(np.argmin(rms))
        if rms[j] < 1.2 and straightness(c, ks[j])[2] > 350 and (rms.max() - rms[j]) > 1.0 and 0 < j < len(ks) - 1:
            keep.append(c)
    return keep


def main(out_path, fixed_k1=None):
    data = {c: load(c) for c in CAMS}
    plumbs = {c: plumb_chains(c) for c in CAMS}
    prev = json.load(open('data/calib_final.json'))
    x2 = prev['cam2']
    k1_0 = fixed_k1 if fixed_k1 is not None else -0.24
    f0 = np.array(x2['K'])[0, 0]
    shared = np.r_[f0, k1_0, 0.0, x2['rho']]
    poses, flips = {}, {}
    for c in CAMS:
        flip, xi = cf.init(data[c]['seed'])
        flips[c] = flip
        K = np.array([[shared[0], 0, W / 2], [0, shared[0], H / 2], [0, 0, 1]])
        obj = cf.seed_world(data[c]['seed'], shared[3], flip)
        img = np.float32([data[c]['seed'][k] for k in data[c]['seed']])
        ok, rv, tv = cv2.solvePnP(obj, img, K, np.array([shared[1], 0, 0, 0, 0]), flags=cv2.SOLVEPNP_IPPE)
        poses[c] = np.r_[rv.ravel(), tv.ravel()]
    x = np.r_[shared, poses['cam1'], poses['cam2']]
    centres = {c: cf.seed_world(data[c]['seed'], shared[3], flips[c])[:, :2].mean(0) for c in CAMS}
    lo = np.r_[0.5 * W, -0.5, -1e-9, 1.5, [-np.inf] * 12]
    hi = np.r_[1.3 * W, 0.2, 1e-9, 2.8, [np.inf] * 12]
    if fixed_k1 is not None:
        lo[1], hi[1] = fixed_k1 - 1e-9, fixed_k1 + 1e-9
    for stage, radius in enumerate((3, 6, 10, 40)):
        sel = {}
        for c in CAMS:
            xc = cam_x(x, c)
            rf, w0, cosax = cf.floor_res(xc, data[c]['pts'], data[c]['along'])
            s = (np.linalg.norm(w0 - centres[c], axis=1) < radius) & (cosax > np.cos(np.radians(10))) & (np.abs(rf) < 5)
            V = data[c]['vsegs'][np.abs(cf.vert_res(xc, data[c]['vsegs'])) < (30 if stage == 0 else 12)]
            sel[c] = (data[c]['pts'][s], data[c]['along'][s], V)

        def fun(z):
            out = []
            for c in CAMS:
                zc = cam_x(z, c)
                P, A, V = sel[c]
                out += [2.0 * cf.seed_res(zc, data[c]['seed'], flips[c]), cf.floor_res(zc, P, A)[0],
                        0.5 * cf.vert_res(zc, V), cf.roll_res(zc), plumb_res(zc, plumbs[c])]
            return np.concatenate(out)
        x = least_squares(fun, x, loss='soft_l1', f_scale=1.5, bounds=(lo, hi), x_scale='jac', max_nfev=800).x
        msg = []
        for c in CAMS:
            zc = cam_x(x, c)
            P, A, V = sel[c]
            rf = cf.floor_res(zc, P, A)[0]; rs = cf.seed_res(zc, data[c]['seed'], flips[c])
            K, dist, rvec, tvec, rho = cf.unpack(zc)
            R, _ = cv2.Rodrigues(rvec); C = -R.T @ tvec
            pr = plumb_res(zc, plumbs[c])
            msg.append('%s floor %d med %.2f p90 %.2f | seeds rms %.2f | plumb rms %.2f (%d chains) | height %.2f m'
                       % (c, len(P), np.median(np.abs(rf)), np.percentile(np.abs(rf), 90), np.sqrt(np.mean(rs ** 2)),
                          np.sqrt(np.mean(pr ** 2)) if len(pr) else 0, len(plumbs[c]), C[2] * 0.26))
        print('stage %d r=%s f=%.0f k1=%.3f rho=%.3f\n   %s\n   %s' % (stage, radius, x[0], x[1], x[3], msg[0], msg[1]))
    out = {}
    for c in CAMS:
        K, dist, rvec, tvec, rho = cf.unpack(cam_x(x, c))
        out[c] = dict(K=K.tolist(), dist=dist.tolist(), rvec=list(map(float, rvec)), tvec=list(map(float, tvec)),
                      rho=float(rho), flip=bool(flips[c]))
    json.dump(out, open(out_path, 'w'), indent=1)


if __name__ == '__main__':
    main(sys.argv[1], float(sys.argv[2]) if len(sys.argv) > 2 else None)
