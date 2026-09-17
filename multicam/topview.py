"""Metric top-down view of the floor from each camera's median background.

Floor frame in metres (1 tile short side = UNIT m, from people heights). Used to
register the two cameras by image alignment of the floor itself, and later as
the base layer of the 2D/3D plan."""
import cv2, json, numpy as np

W, H = 2560, 1440
UNIT = 0.26          # metres per tile short side (people heights: 25.0-26.3 cm)
RES = 0.02           # metres per top-view pixel


def render(cam, calib, extent, A=np.eye(2), T=np.zeros(2)):
    """extent (x0, x1, y0, y1) in metres of the TARGET frame; A, T map this camera's
    floor frame (metres) into the target frame: target = A @ own + T."""
    p = calib[cam]
    K, dist = np.array(p['K']), np.array(p['dist'])
    rv, tv = np.array(p['rvec']), np.array(p['tvec'])
    x0, x1, y0, y1 = extent
    xs = np.arange(x0, x1, RES); ys = np.arange(y1, y0, -RES)
    gx, gy = np.meshgrid(xs, ys)
    tgt = np.stack([gx.ravel(), gy.ravel()], 1)
    own = (np.linalg.inv(A) @ (tgt - T).T).T / UNIT                 # back to this camera's tile units
    wp = np.c_[own, np.zeros(len(own))].astype(np.float32)
    R, _ = cv2.Rodrigues(rv)
    z = (wp @ R.T + tv.reshape(1, 3))[:, 2]
    c = wp @ R.T + tv.reshape(1, 3)
    rn = np.linalg.norm(c[:, :2] / np.maximum(c[:, 2:3], 1e-6), axis=1)
    pp = cv2.projectPoints(wp, rv, tv, K, dist)[0].reshape(-1, 2)
    bad = (z <= 0.3) | (rn > 0.85)
    pp[bad] = -1
    mx = pp[:, 0].reshape(gx.shape).astype(np.float32); my = pp[:, 1].reshape(gx.shape).astype(np.float32)
    img = cv2.imread('data/%s_bg.jpg' % cam)
    out = cv2.remap(img, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    from rooms import mask
    m = cv2.remap(mask(cam), mx, my, cv2.INTER_NEAREST, borderValue=0)
    valid = (mx >= 0) & (mx < W) & (my >= 0) & (my < H)
    return out, m, valid


if __name__ == '__main__':
    calib = json.load(open('data/calib_final.json'))
    for cam in ('cam1', 'cam2'):
        p = calib[cam]
        R, _ = cv2.Rodrigues(np.array(p['rvec'])); C = -R.T @ np.array(p['tvec'])
        cx, cy = C[0] * UNIT, C[1] * UNIT
        ext = (cx - 9, cx + 9, cy - 2, cy + 16) if cam == 'cam2' else (cx - 9, cx + 9, cy - 2, cy + 16)
        out, m, valid = render(cam, calib, ext)
        print(cam, 'camera at (%.2f, %.2f) m, height %.2f m' % (cx, cy, C[2] * UNIT))
        cv2.imwrite('data/%s_top_metric.jpg' % cam, cv2.resize(out, None, fx=0.5, fy=0.5))
