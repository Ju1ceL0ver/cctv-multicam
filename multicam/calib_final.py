"""Camera calibration from the shop's own floor: rectangular tiles + verticals.

The floor tiles are rectangles (aspect ~1:2), not squares -- every square-grid
model contradicted itself until that was established: the seed homography gives
the same focal length from perpendicularity on both cameras (~2160 px) while
admitting no solution with equal tile sides.

World frame: floor plane z = 0, X along the short tile side (1 unit), Y along
the long side (rho units, estimated). Residuals, all in pixels:
  * hand-read seam crossings, reprojected with their lattice indices;
  * every ridge point of a floor seam, back-projected: X mod 1 or Y mod rho;
  * vertical edges off the floor pointing at the vertical vanishing point.
The region of trusted floor points widens in stages from the seed block."""
import cv2, json, numpy as np
from scipy.optimize import least_squares
from rooms import mask
from lines import segments
from wrapfit import ridges
from grow import MANUAL_SEED

W, H = 2560, 1440


def unpack(x):
    f, k1, k2, rho = x[0], x[1], x[2], x[3]
    K = np.array([[f, 0, W / 2], [0, f, H / 2], [0, 0, 1]])
    return K, np.array([k1, k2, 0, 0, 0]), x[4:7], x[7:10], rho


def seed_world(seed, rho, flip):
    return np.float32([(c, (-r if flip else r) * rho, 0) for (r, c) in seed])


def backproject(px, K, dist, rv, tv):
    n = cv2.undistortPoints(px.reshape(-1, 1, 2), K, dist).reshape(-1, 2)
    R, _ = cv2.Rodrigues(rv)
    C = -R.T @ tv
    rays = np.c_[n, np.ones(len(n))] @ R
    lam = -C[2] / rays[:, 2]
    return C[None, :2] + lam[:, None] * rays[:, :2], lam


def seed_res(x, seed, flip):
    K, dist, rv, tv, rho = unpack(x)
    obj = seed_world(seed, rho, flip)
    img = np.float32([seed[k] for k in seed])
    return (cv2.projectPoints(obj, rv, tv, K, dist)[0].reshape(-1, 2) - img).ravel()


def floor_res(x, pts, along):
    K, dist, rv, tv, rho = unpack(x)
    w0, l0 = backproject(pts, K, dist, rv, tv)
    w1, _ = backproject(pts + 2.0 * along, K, dist, rv, tv)
    d = w1 - w0
    dn = np.linalg.norm(d, axis=1) + 1e-12
    cosax = np.maximum(np.abs(d[:, 0]), np.abs(d[:, 1])) / dn
    along_y = np.abs(d[:, 1]) > np.abs(d[:, 0])        # runs along Y -> it is a line X = n
    frac = np.where(along_y, w0[:, 0] - np.round(w0[:, 0]), w0[:, 1] / rho - np.round(w0[:, 1] / rho))
    period = np.where(along_y, 1.0, rho)
    r = frac * period * (2.0 / dn)
    bad = ~np.isfinite(r) | (l0 <= 0)
    r[bad] = 10.0
    cosax[bad] = 0
    return r, w0, cosax


def vert_res(x, vsegs):
    if len(vsegs) == 0:
        return np.zeros(0)
    K, dist, rv, tv, rho = unpack(x)
    R, _ = cv2.Rodrigues(rv)
    p1 = cv2.undistortPoints(vsegs[:, :2].reshape(-1, 1, 2), K, dist, P=K).reshape(-1, 2)
    p2 = cv2.undistortPoints(vsegs[:, 2:].reshape(-1, 1, 2), K, dist, P=K).reshape(-1, 2)
    m, d = (p1 + p2) / 2, p2 - p1
    L = np.linalg.norm(d, axis=1) + 1e-9
    d /= L[:, None]
    v = K @ R[:, 2]
    u = v[:2][None, :] - m * v[2]
    u /= np.linalg.norm(u, axis=1)[:, None] + 1e-12
    return (d[:, 0] * u[:, 1] - d[:, 1] * u[:, 0]) * L / 2


def roll_res(x, scale=4.0):
    """Soft prior: CCTV cameras are mounted close to level (cam2 solved to -0.1 deg on its own)."""
    K, dist, rv, tv, rho = unpack(x)
    R, _ = cv2.Rodrigues(rv)
    right = R.T @ np.array([1.0, 0, 0])
    return np.array([np.degrees(np.arcsin(np.clip(right[2], -1, 1))) / scale * 10.0])


def init(seed):
    src = np.float32([(c, r) for (r, c) in seed]); dst = np.float32([seed[k] for k in seed])
    Hm, _ = cv2.findHomography(src, dst, 0)
    T = np.array([[1, 0, -W / 2], [0, 1, -H / 2], [0, 0, 1]])
    h1, h2 = (T @ Hm)[:, 0], (T @ Hm)[:, 1]
    f = np.sqrt(-(h1[0] * h2[0] + h1[1] * h2[1]) / (h1[2] * h2[2]))
    K = np.array([[f, 0, W / 2], [0, f, H / 2], [0, 0, 1]])
    M = np.linalg.inv(K) @ Hm
    rho = np.linalg.norm(M[:, 1]) / np.linalg.norm(M[:, 0])
    best = None
    for flip in (False, True):
        obj = seed_world(seed, rho, flip)
        ok, rv, tv = cv2.solvePnP(obj, dst, K, np.zeros(5), flags=cv2.SOLVEPNP_IPPE)
        R, _ = cv2.Rodrigues(rv)
        C = -R.T @ tv.ravel()
        if C[2] > 0 and (best is None):
            best = (flip, np.r_[f, 0.0, 0.0, rho, rv.ravel(), tv.ravel()])
    return best


def calibrate(cam):
    img = cv2.imread('data/%s_bg.jpg' % cam)
    fm = mask(cam)
    pts, along, _ = ridges(img, cv2.erode(fm, np.ones((9, 9), np.uint8)))
    segs = segments(img, min_len=30)
    mid = ((segs[:, :2] + segs[:, 2:]) / 2).astype(int)
    inside = fm[np.clip(mid[:, 1], 0, H - 1), np.clip(mid[:, 0], 0, W - 1)] > 0
    L = np.hypot(segs[:, 2] - segs[:, 0], segs[:, 3] - segs[:, 1])
    ang = np.degrees(np.arctan2(np.abs(segs[:, 3] - segs[:, 1]), np.abs(segs[:, 2] - segs[:, 0])))
    vsegs = segs[(~inside) & (L >= 90) & (ang > 50)]
    seed = MANUAL_SEED[cam]
    flip, x = init(seed)
    print('%s: init f=%.0f rho=%.2f flip=%s | seed rms %.2f px' % (cam, x[0], x[3], flip,
          np.sqrt(np.mean(seed_res(x, seed, flip) ** 2))))
    K, dist, rv, tv, rho = unpack(x)
    centre = seed_world(seed, rho, flip)[:, :2].mean(0)
    lo = [0.4 * W, -0.6, -1e-9, 1.0] + [-np.inf] * 6
    hi = [1.5 * W, 0.3, 1e-9, 3.5] + [np.inf] * 6
    for stage, (radius, free_k2, use_vert) in enumerate(((2.5, False, True), (5, False, True), (8, False, True),
                                                        (14, False, True), (40, False, True), (40, True, True))):
        rf, w0, cosax = floor_res(x, pts, along)
        sel = (np.linalg.norm(w0 - centre, axis=1) < radius) & (cosax > np.cos(np.radians(10))) & (np.abs(rf) < 5)
        P, A = pts[sel], along[sel]
        V = vsegs[np.abs(vert_res(x, vsegs)) < (40 if stage == 0 else 12)] if use_vert else vsegs[:0]
        lo[2], hi[2] = ((-0.5, 0.5) if free_k2 else (-1e-9, 1e-9))
        x[2] = np.clip(x[2], lo[2], hi[2])
        fun = lambda z: np.concatenate([2.0 * seed_res(z, seed, flip), floor_res(z, P, A)[0], 0.5 * vert_res(z, V),
                                        roll_res(z)])
        x = least_squares(fun, x, loss='soft_l1', f_scale=1.5, bounds=(lo, hi), x_scale='jac', max_nfev=500).x
        rf = floor_res(x, P, A)[0]; rs = seed_res(x, seed, flip); rvv = vert_res(x, V) if len(V) else np.zeros(1)
        K, dist, rv, tv, rho = unpack(x)
        R, _ = cv2.Rodrigues(rv); C = -R.T @ tv
        print('  stage %d r=%-4s floor %4d pts med %.2f p90 %.2f px | seeds rms %.2f px | vertical %3d med %.2f px | '
              'f=%.0f k1=%.3f k2=%.3f rho=%.3f height=%.2f units roll=%.1f'
              % (stage, radius, len(P), np.median(np.abs(rf)), np.percentile(np.abs(rf), 90),
                 np.sqrt(np.mean(rs ** 2)), len(V), np.median(np.abs(rvv)), x[0], x[1], x[2], rho, C[2], roll_res(x)[0] * 0.4))
    return img, x, flip


def draw(cam, img, x):
    K, dist, rv, tv, rho = unpack(x)
    R, _ = cv2.Rodrigues(rv); C = -R.T @ tv
    vis = (img * 0.6).astype(np.uint8)
    for n in range(int(C[0]) - 30, int(C[0]) + 31):
        for axis in (0, 1):
            t = np.linspace(-40, 40, 1600)
            wp = np.float32([(n, C[1] + tt, 0) for tt in t]) if axis == 0 else \
                np.float32([(C[0] + tt, n * rho, 0) for tt in t])
            zc = (wp @ R.T + tv.reshape(1, 3))[:, 2]
            wp = wp[zc > 0.5]
            if len(wp) < 2:
                continue
            pp = cv2.projectPoints(wp, rv, tv, K, dist)[0].reshape(-1, 2)
            ok = (pp[:, 0] > -50) & (pp[:, 0] < W + 50) & (pp[:, 1] > -50) & (pp[:, 1] < H + 50)
            idx = np.nonzero(ok)[0]
            if len(idx) < 2:
                continue
            for run in np.split(idx, np.nonzero(np.diff(idx) > 1)[0] + 1):
                if len(run) > 1:
                    cv2.polylines(vis, [pp[run].astype(np.int32)], False, (0, 220, 255) if axis == 0 else (255, 90, 255), 2)
    cv2.imwrite('data/%s_final.jpg' % cam, vis)
    cv2.imwrite('data/%s_final_small.jpg' % cam, cv2.resize(vis, (1280, 720)))


if __name__ == '__main__':
    import sys
    out = {}
    for cam in sys.argv[1:] or ('cam2', 'cam1'):
        img, x, flip = calibrate(cam)
        draw(cam, img, x)
        K, dist, rv, tv, rho = unpack(x)
        out[cam] = dict(K=K.tolist(), dist=dist.tolist(), rvec=list(map(float, rv)), tvec=list(map(float, tv)),
                        rho=float(rho), flip=bool(flip))
    json.dump(out, open('data/calib_final.json', 'w'), indent=1)
