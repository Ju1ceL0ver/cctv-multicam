"""Tile lattice: Hough seed in a clean patch, then neighbour-driven growth.

Every node is predicted from already-found nodes around it with a LOCAL
homography (so lens distortion never has to be modelled during growth), then
snapped to the true seam crossing. Only after the lattice is grown is the full
camera model solved."""
import cv2, json, numpy as np
from seams import seam_response
from rooms import mask
from lattice import cross_template, snap

W, H = 2560, 1440
SEED = {'cam1': (950, 1100, 1450, 1440), 'cam2': (950, 1050, 1370, 1440)}
# Seam crossings read off 2-3x zoomed crops of the median background; (row, col)
# are lattice indices along the two seam families. Snapping refines them.
MANUAL_SEED = {
    'cam1': {(-1, 1): (924, 1152.5), (-1, 2): (990, 1108), (-1, 3): (1061.5, 1060),
             (0, 0): (970, 1295), (0, 1): (1045, 1237.5), (0, 2): (1120, 1187.5), (0, 3): (1190, 1134),
             (1, 0): (1109, 1400), (1, 1): (1190, 1339), (1, 2): (1264, 1276), (1, 3): (1337.5, 1219)},
    'cam2': {(1, 1): (1070.0, 1013.3), (1, 2): (1146.7, 986.7), (2, 0): (1076.7, 1141.7), (2, 1): (1155.0, 1110.7),
             (2, 2): (1240.0, 1078.3), (3, 0): (1173.3, 1253.3), (3, 1): (1263.3, 1216.7), (3, 2): (1343.3, 1178.3)},
}


def hough_seed(resp, fmask, rect):
    x0, y0, x1, y1 = rect
    r = resp[y0:y1, x0:x1] * (fmask[y0:y1, x0:x1] > 0)
    thr = np.percentile(r[r > 0], 93)
    b = (r > thr).astype(np.uint8) * 255
    lines = cv2.HoughLines(b, 2, np.radians(0.5), int(0.25 * min(b.shape)))
    if lines is None:
        return []
    lines = lines[:, 0, :]
    # merge duplicates
    kept = []
    for rho, th in lines:
        if all(abs(rho - r2) > 14 or abs(np.sin(th - t2)) > np.sin(np.radians(4)) for r2, t2 in kept):
            kept.append((rho, th))
        if len(kept) >= 16:
            break
    kept = np.array(kept)
    # two families by angle (doubled-angle k-means)
    z = np.stack([np.cos(2 * kept[:, 1]), np.sin(2 * kept[:, 1])], 1).astype(np.float32)
    _, lab, _ = cv2.kmeans(z, 2, None, (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 1e-4), 5,
                           cv2.KMEANS_PP_CENTERS)
    fams = [kept[lab.ravel() == k] for k in range(2)]
    for k in range(2):
        f = fams[k]
        # orient consistently, then order by signed distance
        th0 = f[0, 1]
        rh = np.where(np.cos(f[:, 1] - th0) < 0, -f[:, 0], f[:, 0])
        fams[k] = f[np.argsort(rh)]
    nodes = {}
    for i, (ra, ta) in enumerate(fams[0]):
        for j, (rb, tb) in enumerate(fams[1]):
            A = np.array([[np.cos(ta), np.sin(ta)], [np.cos(tb), np.sin(tb)]])
            if abs(np.linalg.det(A)) < 0.2:
                continue
            x, y = np.linalg.solve(A, [ra, rb])
            if 0 <= x < b.shape[1] and 0 <= y < b.shape[0]:
                nodes[(i, j)] = (x + x0, y + y0)
    return nodes


def local_predict(found, ij, k=10):
    keys = np.array(list(found.keys()), float)
    d = np.abs(keys - np.array(ij, float)).max(1)
    order = np.argsort(d)
    near = [tuple(keys[o].astype(int)) for o in order[:k] if d[o] <= 2]
    # a homography extrapolated from one or two collinear rows is how a whole row
    # of nodes ends up one tile off: demand real 2D support
    if len(near) < 5 or len({n[0] for n in near}) < 2 or len({n[1] for n in near}) < 2:
        return None
    src = np.float32([n for n in near])
    dst = np.float32([found[n] for n in near])
    # need non-collinear support
    if np.linalg.matrix_rank(np.c_[src, np.ones(len(src))]) < 3:
        return None
    Hm, _ = cv2.findHomography(src, dst, 0)
    if Hm is None:
        return None
    def at(p):
        v = Hm @ np.array([p[0], p[1], 1.0])
        return v[:2] / v[2]
    c = at(ij)
    return c, at((ij[0] + 0.3, ij[1])) - c, at((ij[0], ij[1] + 0.3)) - c


def grow(cam, max_rounds=40):
    img = cv2.imread('data/%s_bg.jpg' % cam)
    resp = seam_response(img).astype(np.float32)
    fmask = cv2.erode(mask(cam), np.ones((7, 7), np.uint8))
    seed = MANUAL_SEED[cam]
    found = {}
    for ij, (x, y) in seed.items():
        # snap each Hough intersection; orientation from neighbours in the seed itself
        nb_a = seed.get((ij[0] + 1, ij[1])) or seed.get((ij[0] - 1, ij[1]))
        nb_b = seed.get((ij[0], ij[1] + 1)) or seed.get((ij[0], ij[1] - 1))
        if nb_a is None or nb_b is None:
            continue
        da = np.array(nb_a) - (x, y); db = np.array(nb_b) - (x, y)
        size = min(np.linalg.norm(da), np.linalg.norm(db))
        s = snap(resp, fmask, [(x, y)], [db], [da], [size], radius_frac=0.15, min_ncc=0.30)[0]
        if s is not None:
            found[ij] = (float(s[0]), float(s[1]))
        else:
            print('  seed node', ij, 'did not snap')
    print(cam, 'seed nodes: %d hough -> %d snapped' % (len(seed), len(found)))
    rejected = set()
    for rnd in range(max_rounds):
        cand = set()
        for (i, j) in found:
            for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                n = (i + di, j + dj)
                if n not in found and n not in rejected:
                    cand.add(n)
        added = 0
        for n in sorted(cand):
            pr = local_predict(found, n)
            if pr is None:
                continue
            c, da, db = pr
            size = min(np.linalg.norm(da), np.linalg.norm(db)) / 0.3
            s = snap(resp, fmask, [c], [db], [da], [size], radius_frac=0.07, min_ncc=0.35)[0]
            if s is None:
                rejected.add(n)
                continue
            if np.hypot(s[0] - c[0], s[1] - c[1]) > 0.08 * size:
                rejected.add(n); continue
            # perspective changes gradually: the step onto the new node must continue
            # the step that led to its neighbour along the same lattice direction
            ok_step = True
            for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                m = found.get((n[0] + di, n[1] + dj)); mm = found.get((n[0] + 2 * di, n[1] + 2 * dj))
                if m is not None and mm is not None:
                    ratio = np.hypot(s[0] - m[0], s[1] - m[1]) / max(1e-6, np.hypot(m[0] - mm[0], m[1] - mm[1]))
                    if not (0.6 <= ratio <= 1.25):
                        ok_step = False
            if not ok_step:
                rejected.add(n); continue
            found[n] = (float(s[0]), float(s[1])); added += 1
        if added == 0:
            break
    print('  grown to %d nodes in %d rounds' % (len(found), rnd + 1))
    return img, found


if __name__ == '__main__':
    import sys
    out = {}
    for cam in sys.argv[1:] or ('cam1', 'cam2'):
        img, found = grow(cam)
        vis = (img * 0.5).astype(np.uint8)
        x0, y0, x1, y1 = SEED[cam]
        cv2.rectangle(vis, (x0, y0), (x1, y1), (255, 255, 0), 3)
        for (i, j), (x, y) in found.items():
            cv2.circle(vis, (int(x), int(y)), 10, (0, 255, 0), 3)
        for (i, j), (x, y) in found.items():
            for di, dj in ((1, 0), (0, 1)):
                m = found.get((i + di, j + dj))
                if m:
                    cv2.line(vis, (int(x), int(y)), (int(m[0]), int(m[1])), (255, 80, 255) if di else (0, 200, 255), 2)
        cv2.imwrite('data/%s_grow.jpg' % cam, cv2.resize(vis, (1280, 720)))
        out[cam] = {'%d,%d' % k: v for k, v in found.items()}
    json.dump(out, open('data/grow.json', 'w'))
