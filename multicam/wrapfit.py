"""Camera calibration against the tile grid with no node detection and no lattice
indexing.

Every ridge pixel of a thin dark line on the floor (Hessian analysis gives its
position and direction) is back-projected onto the floor in tile units. A seam
must lie ON an integer grid line (X = n or Y = n) and run ALONG it; which n is
irrelevant, so residuals are taken modulo one tile. There is nothing to index,
so an off-by-one-tile error cannot exist. Scratches do not run along the grid
axes and drop out on the direction test.

Starts from eight hand-read crossings near the image centre and widens the
region in stages, so the modulo assignment is only trusted where the model is
already better than a third of a tile."""
import cv2, json, numpy as np
from scipy.optimize import least_squares
from rooms import mask
from grow import MANUAL_SEED

W, H = 2560, 1440


def ridges(img, fmask, sigma=1.8, stride=3):
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    g = cv2.GaussianBlur(g, (0, 0), sigma)
    dxx = cv2.Sobel(g, cv2.CV_32F, 2, 0, ksize=3)
    dyy = cv2.Sobel(g, cv2.CV_32F, 0, 2, ksize=3)
    dxy = cv2.Sobel(g, cv2.CV_32F, 1, 1, ksize=3)
    tr = dxx + dyy
    disc = np.sqrt(((dxx - dyy) / 2) ** 2 + dxy ** 2)
    l1 = tr / 2 + disc                       # larger eigenvalue: across a dark line it is positive
    thr = np.percentile(l1[fmask > 0], 96)
    ys, xs = np.nonzero((l1 > thr) & (fmask > 0))
    sel = (xs % stride == 0) & (ys % stride == 0)
    xs, ys = xs[sel], ys[sel]
    # eigenvector of the larger eigenvalue is ACROSS the line; the line runs perpendicular
    a, b, c = dxx[ys, xs], dxy[ys, xs], dyy[ys, xs]
    lam = l1[ys, xs]
    vx, vy = b, lam - a
    nrm = np.hypot(vx, vy) + 1e-9
    across = np.stack([vx / nrm, vy / nrm], 1)
    along = np.stack([-across[:, 1], across[:, 0]], 1)
    return np.stack([xs, ys], 1).astype(np.float64), along, lam


def unpack(x):
    f, k1, k2 = x[0], x[1], x[2]
    K = np.array([[f, 0, W / 2], [0, f, H / 2], [0, 0, 1]])
    return K, np.array([k1, k2, 0, 0, 0]), x[3:6], x[6:9]


def to_floor(px, K, dist, rv, tv):
    nrm = cv2.undistortPoints(px.reshape(-1, 1, 2), K, dist).reshape(-1, 2)
    R, _ = cv2.Rodrigues(rv)
    C = -R.T @ tv
    rays = np.c_[nrm, np.ones(len(nrm))] @ R
    lam = -C[2] / rays[:, 2]
    return C[None, :2] + lam[:, None] * rays[:, :2], lam


def residuals(x, pts, along, fixed_k2):
    K, dist, rv, tv = unpack(x)
    if fixed_k2:
        dist[1] = 0.0
    w0, l0 = to_floor(pts, K, dist, rv, tv)
    w1, _ = to_floor(pts + 2.0 * along, K, dist, rv, tv)
    d = w1 - w0
    dn = np.linalg.norm(d, axis=1) + 1e-9
    px_per_tile = 2.0 / dn                                 # image px per tile along the seam
    along_y = np.abs(d[:, 1]) > np.abs(d[:, 0])            # seam runs along Y -> constant X
    coord = np.where(along_y, w0[:, 0], w0[:, 1])
    r = (coord - np.round(coord)) * px_per_tile
    bad = ~np.isfinite(r) | (l0 < 0)
    r[bad] = 10.0
    return r, along_y, d / dn[:, None], w0


def fit_cam(cam):
    img = cv2.imread('data/%s_bg.jpg' % cam)
    fmask = cv2.erode(mask(cam), np.ones((9, 9), np.uint8))
    pts, along, strength = ridges(img, fmask)
    seed = MANUAL_SEED[cam]
    obj = np.float32([(c, r, 0) for (r, c) in seed])
    ipt = np.float32([seed[k] for k in seed])
    best = None
    for f0 in np.linspace(0.35, 1.0, 14) * W:
        for k10 in (-0.05, -0.15, -0.3):
            K0 = np.array([[f0, 0, W / 2], [0, f0, H / 2], [0, 0, 1]]); d0 = np.array([k10, 0, 0, 0, 0])
            ok, rv, tv = cv2.solvePnP(obj, ipt, K0, d0, flags=cv2.SOLVEPNP_ITERATIVE)
            if not ok:
                continue
            e = np.linalg.norm(cv2.projectPoints(obj, rv, tv, K0, d0)[0].reshape(-1, 2) - ipt, axis=1).mean()
            x = np.r_[f0, k10, 0.0, rv.ravel(), tv.ravel()]
            if tv.ravel()[2] <= 0:
                continue
            # score by how many nearby ridge points already sit on grid lines
            r, along_y, dirs, w0 = residuals(x, pts, along, True)
            near = np.linalg.norm(w0 - obj[:, :2].mean(0), axis=1) < 4
            axis_ok = np.maximum(np.abs(dirs[:, 0]), np.abs(dirs[:, 1])) > np.cos(np.radians(10))
            sc = np.mean(np.abs(r[near & axis_ok]) < 2.0) if (near & axis_ok).sum() > 50 else 0
            if best is None or sc > best[0]:
                best = (sc, x, e)
    sc, x, e = best
    centre = obj[:, :2].mean(0)
    print('%s: %d ridge points; init f=%.0f k1=%.2f, seed reproj %.1f px, on-grid share %.2f'
          % (cam, len(pts), x[0], x[1], e, sc))
    for radius, fixed_k2 in ((3, True), (5, True), (8, True), (12, True), (40, True), (40, False), (40, False)):
        r, along_y, dirs, w0 = residuals(x, pts, along, fixed_k2)
        axis_ok = np.maximum(np.abs(dirs[:, 0]), np.abs(dirs[:, 1])) > np.cos(np.radians(12))
        sel = (np.linalg.norm(w0 - centre, axis=1) < radius) & axis_ok & (np.abs(r) < 6)
        P, A = pts[sel], along[sel]
        lo = [0.3 * W, -1.0, -1.0 if not fixed_k2 else -1e-9, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf]
        hi = [2.0 * W, 0.6, 1.0 if not fixed_k2 else 1e-9, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf]
        x[2] = np.clip(x[2], lo[2], hi[2])
        sol = least_squares(lambda z: residuals(z, P, A, fixed_k2)[0], x, loss='soft_l1', f_scale=1.0,
                            bounds=(lo, hi), x_scale='jac', max_nfev=300)
        x = sol.x
        rr = residuals(x, P, A, fixed_k2)[0]
        K, dist, rv, tv = unpack(x)
        R, _ = cv2.Rodrigues(rv)
        C = -R.T @ tv
        print('  radius %2d tiles: %5d pts, |res| median %.2f px p90 %.2f | f=%.0f k1=%.4f k2=%.4f height %.2f tiles'
              % (radius, len(P), np.median(np.abs(rr)), np.percentile(np.abs(rr), 90), x[0], x[1], x[2], C[2]))
    # overlay: projected grid lines
    K, dist, rv, tv = unpack(x)
    R, _ = cv2.Rodrigues(rv)
    C = -R.T @ tv
    vis = (img * 0.55).astype(np.uint8)
    for n in range(int(C[0]) - 30, int(C[0]) + 31):
        for axis in (0, 1):
            t = np.linspace(-30, 30, 600)
            wp = np.float32([(n, C[1] + tt, 0) if axis == 0 else (C[0] + tt, n, 0) for tt in t])
            zc = (wp @ R.T + tv.reshape(1, 3))[:, 2]
            wp = wp[zc > 0.4]
            if len(wp) < 2:
                continue
            pp = cv2.projectPoints(wp, rv, tv, K, dist)[0].reshape(-1, 2)
            inside = (pp[:, 0] > -500) & (pp[:, 0] < W + 500) & (pp[:, 1] > -500) & (pp[:, 1] < H + 500)
            pp = pp[inside]
            if len(pp) > 1:
                cv2.polylines(vis, [pp.astype(np.int32)], False, (0, 220, 255) if axis == 0 else (255, 90, 255), 2)
    cv2.imwrite('data/%s_wrap.jpg' % cam, vis)
    cv2.imwrite('data/%s_wrap_small.jpg' % cam, cv2.resize(vis, (1280, 720)))
    return dict(f=float(x[0]), k1=float(x[1]), k2=float(x[2]), rvec=list(map(float, rv)), tvec=list(map(float, tv)),
                height_tiles=float(C[2]), cam_xy_tiles=[float(C[0]), float(C[1])])


if __name__ == '__main__':
    import sys
    out = {}
    for cam in sys.argv[1:] or ('cam2', 'cam1'):
        out[cam] = fit_cam(cam)
    json.dump(out, open('data/wrapfit.json', 'w'), indent=1)
