"""A detected person as a vertical segment standing on the calibrated floor.

World frame is each camera's own floor frame in tile units (1 unit = UNIT m,
z up, floor z = 0). From the foot pixel the floor point is exact up to the
foot-pixel error; from the head pixel we get the person's height when the feet
are visible, and the floor point from a known height when they are not -- the
usual case in this shop, where display stands cut people off at the knees."""
import cv2, json, numpy as np
from topview import UNIT
from rooms import mask

W, H = 2560, 1440


class Camera:
    def __init__(self, cam, calib):
        p = calib[cam]
        self.name = cam
        self.K, self.dist = np.array(p['K']), np.array(p['dist'])
        self.R, _ = cv2.Rodrigues(np.array(p['rvec']))
        self.t = np.array(p['tvec'])
        self.C = -self.R.T @ self.t                       # camera centre, tile units
        m = mask(cam)
        # a foot counts as visible only well inside traced open floor
        self.floor = cv2.erode(m, np.ones((25, 25), np.uint8))

    def rays(self, px):
        n = cv2.undistortPoints(np.asarray(px, np.float64).reshape(-1, 1, 2), self.K, self.dist).reshape(-1, 2)
        d = np.c_[n, np.ones(len(n))] @ self.R            # world directions
        return d

    def on_plane(self, px, z=0.0):
        d = self.rays(px)
        lam = (z - self.C[2]) / d[:, 2]
        P = self.C[None] + lam[:, None] * d
        P[lam <= 0] = np.nan
        return P[:, :2]

    def height(self, foot_px, head_px):
        """Height (m) of the vertical line above each floor point that best meets the head ray."""
        F = self.on_plane(foot_px)
        d = self.rays(head_px)
        out = np.full(len(F), np.nan)
        for i in range(len(F)):
            if not np.all(np.isfinite(F[i])):
                continue
            # closest approach between line A: (F, 0) + s*(0,0,1) and ray B: C + r*d
            a = np.array([0, 0, 1.0]); b = d[i] / np.linalg.norm(d[i])
            w0 = np.r_[F[i], 0.0] - self.C
            A_ = np.array([[a @ a, -a @ b], [a @ b, -b @ b]])
            rhs = np.array([-a @ w0, -b @ w0])
            try:
                s, r = np.linalg.solve(A_, rhs)
            except np.linalg.LinAlgError:
                continue
            out[i] = s * UNIT
        return out

    def foot_visible(self, foot_px, box):
        x, y = int(round(foot_px[0])), int(round(foot_px[1]))
        if not (0 <= x < W and 0 <= y < H) or box[3] > H - 40 or box[1] < 8:
            return False
        return self.floor[y, x] > 0


def map_cam1(xy, reg):
    """cam1 floor (metres, own frame) -> common (cam2) frame."""
    if reg.get('model') == 'affine':
        M = np.array(reg['M'])
        return xy @ M[:, :2].T + M[:, 2]
    A = np.eye(2) if reg.get('rotation_deg', 0) == 0 else -np.eye(2)
    return (A @ xy.T).T + np.array(reg['T_m'])


def place(cams, reg, dets, prior_h=1.68):
    """Per detection: floor xy in the COMMON frame (metres), measured height or nan,
    whether the foot was used. dets: (t, x1, y1, x2, y2, score, fx, fy, hx, hy)."""
    A = np.eye(2) if reg.get('rotation_deg', 0) == 0 else -np.eye(2)
    out = {}
    for cam, d in dets.items():
        c = cams[cam]
        foot, head = d[:, 6:8], d[:, 8:10]
        vis = np.array([c.foot_visible(f, b) for f, b in zip(foot, d[:, 1:5])])
        xy_foot = c.on_plane(foot) * UNIT
        h = np.where(vis, c.height(foot, head), np.nan)
        xy_head = c.on_plane(head, z=prior_h / UNIT) * UNIT
        xy = np.where(vis[:, None], xy_foot, xy_head)
        if cam == 'cam1':
            xy = map_cam1(xy, reg)
            xy_head = map_cam1(xy_head, reg)
        out[cam] = dict(xy=xy, h=h, vis=vis, xy_head=xy_head)
    return out


if __name__ == '__main__':
    calib = json.load(open('data/calib_final.json'))
    reg = {'rotation_deg': 0, 'T_m': [0.54, 7.61]}
    cams = {c: Camera(c, calib) for c in ('cam1', 'cam2')}
    for c in cams.values():
        print(c.name, 'camera centre (m): x %.2f y %.2f height %.2f' % tuple(c.C * UNIT))
    z = dict(np.load('data/clip/dets25_smoke.npz'))
    dets = {c: z[c] for c in ('cam1', 'cam2')}
    P = place(cams, reg, dets)
    for cam in ('cam1', 'cam2'):
        v = P[cam]['vis']; h = P[cam]['h']
        print(cam, 'detections %d, foot visible %d, heights (m) where visible: %s'
              % (len(v), v.sum(), np.round(np.sort(h[v]), 2)[:40]))
