"""Joint calibration of both cameras: one lens model and one tile, two poses.

Both cameras are the same model (identical stream parameters), so they share
focal length and distortion; they look at the same tiles, so they share the tile
aspect ratio. cam2 is well conditioned on its own (long aisle, clear seams); the
shared parameters carry that into cam1, whose own floor view is short and
diagonal."""
import cv2, json, numpy as np
from scipy.optimize import least_squares
from rooms import mask
from lines import segments
from wrapfit import ridges
from grow import MANUAL_SEED
import calib_final as cf

W, H = 2560, 1440
CAMS = ('cam1', 'cam2')


def split_x(x):
    shared = x[:4]
    poses = {c: x[4 + 6 * i:10 + 6 * i] for i, c in enumerate(CAMS)}
    return shared, poses


def cam_x(x, cam):
    shared, poses = split_x(x)
    return np.r_[shared, poses[cam]]


def load(cam):
    img = cv2.imread('data/%s_bg.jpg' % cam)
    fm = mask(cam)
    pts, along, strength = ridges(img, cv2.erode(fm, np.ones((9, 9), np.uint8)))
    # display-stand edges are far stronger than tile seams: keep the seam-like ridges
    keep = strength < 3.0 * np.median(strength)
    pts, along = pts[keep], along[keep]
    segs = segments(img, min_len=30)
    mid = ((segs[:, :2] + segs[:, 2:]) / 2).astype(int)
    inside = fm[np.clip(mid[:, 1], 0, H - 1), np.clip(mid[:, 0], 0, W - 1)] > 0
    L = np.hypot(segs[:, 2] - segs[:, 0], segs[:, 3] - segs[:, 1])
    ang = np.degrees(np.arctan2(np.abs(segs[:, 3] - segs[:, 1]), np.abs(segs[:, 2] - segs[:, 0])))
    return dict(pts=pts, along=along, vsegs=segs[(~inside) & (L >= 90) & (ang > 50)], seed=MANUAL_SEED[cam])


def main():
    data = {c: load(c) for c in CAMS}
    prev = json.load(open('data/calib_final.json'))
    x2 = prev['cam2']
    shared = np.r_[np.array(x2['K'])[0, 0], x2['dist'][0], 0.0, x2['rho']]
    poses, flips = {}, {}
    for c in CAMS:
        flip, xi = cf.init(data[c]['seed'])
        flips[c] = flip
        # re-solve the pose with the shared lens and tile
        K = np.array([[shared[0], 0, W / 2], [0, shared[0], H / 2], [0, 0, 1]])
        obj = cf.seed_world(data[c]['seed'], shared[3], flip)
        img = np.float32([data[c]['seed'][k] for k in data[c]['seed']])
        ok, rv, tv = cv2.solvePnP(obj, img, K, np.array([shared[1], 0, 0, 0, 0]), flags=cv2.SOLVEPNP_IPPE)
        poses[c] = np.r_[rv.ravel(), tv.ravel()]
    x = np.r_[shared, poses['cam1'], poses['cam2']]
    centres = {c: cf.seed_world(data[c]['seed'], shared[3], flips[c])[:, :2].mean(0) for c in CAMS}
    lo = np.r_[0.5 * W, -0.5, -1e-9, 1.5, [-np.inf] * 12]
    hi = np.r_[1.3 * W, 0.2, 1e-9, 2.8, [np.inf] * 12]
    for stage, (radius, free_k2) in enumerate(((3, False), (6, False), (10, False), (40, False), (40, True))):
        sel = {}
        for c in CAMS:
            xc = cam_x(x, c)
            rf, w0, cosax = cf.floor_res(xc, data[c]['pts'], data[c]['along'])
            s = (np.linalg.norm(w0 - centres[c], axis=1) < radius) & (cosax > np.cos(np.radians(10))) & (np.abs(rf) < 5)
            V = data[c]['vsegs'][np.abs(cf.vert_res(xc, data[c]['vsegs'])) < (30 if stage == 0 else 12)]
            sel[c] = (data[c]['pts'][s], data[c]['along'][s], V)
        lo[2], hi[2] = (-0.4, 0.4) if free_k2 else (-1e-9, 1e-9)
        x[2] = np.clip(x[2], lo[2], hi[2])

        def fun(z):
            out = []
            for c in CAMS:
                zc = cam_x(z, c)
                P, A, V = sel[c]
                out += [2.0 * cf.seed_res(zc, data[c]['seed'], flips[c]), cf.floor_res(zc, P, A)[0],
                        0.5 * cf.vert_res(zc, V), cf.roll_res(zc)]
            return np.concatenate(out)
        x = least_squares(fun, x, loss='soft_l1', f_scale=1.5, bounds=(lo, hi), x_scale='jac', max_nfev=800).x
        msg = []
        for c in CAMS:
            zc = cam_x(x, c)
            P, A, V = sel[c]
            rf = cf.floor_res(zc, P, A)[0]; rs = cf.seed_res(zc, data[c]['seed'], flips[c]); rv = cf.vert_res(zc, V)
            K, dist, rvec, tvec, rho = cf.unpack(zc)
            R, _ = cv2.Rodrigues(rvec); C = -R.T @ tvec
            msg.append('%s floor %d med %.2f p90 %.2f | seeds %.2f | vert %d med %.2f | h=%.2f roll=%.1f'
                       % (c, len(P), np.median(np.abs(rf)), np.percentile(np.abs(rf), 90), np.sqrt(np.mean(rs ** 2)),
                          len(V), np.median(np.abs(rv)), C[2], cf.roll_res(zc)[0] * 0.4))
        print('stage %d r=%s f=%.0f k1=%.3f k2=%.3f rho=%.3f\n   %s\n   %s' % (stage, radius, x[0], x[1], x[2], x[3], msg[0], msg[1]))
    out = {}
    for c in CAMS:
        K, dist, rvec, tvec, rho = cf.unpack(cam_x(x, c))
        out[c] = dict(K=K.tolist(), dist=dist.tolist(), rvec=list(map(float, rvec)), tvec=list(map(float, tvec)),
                      rho=float(rho), flip=bool(flips[c]))
    json.dump(out, open('data/calib_final.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
