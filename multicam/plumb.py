"""Lens distortion from plumb lines: long, physically straight edges in the
median backgrounds (slat walls, glass mullions, stand edges) must come out
straight after undistortion. Each edge chain votes for the k1 that straightens
it; long chains far from the image centre carry the information."""
import cv2, numpy as np

W, H = 2560, 1440
F = 2030.187
K = np.array([[F, 0, W / 2], [0, F, H / 2], [0, 0, 1]])


def chains(img, min_len=350):
    g = cv2.GaussianBlur(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (5, 5), 0)
    e = cv2.Canny(g, 40, 120)
    cnts, _ = cv2.findContours(e, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    out = []
    for c in cnts:
        c = c.reshape(-1, 2).astype(np.float64)
        if len(c) < min_len:
            continue
        # contours of thin edges double back: keep one pass along the principal direction
        mean = c.mean(0); u = np.linalg.svd(c - mean)[2][0]
        s = (c - mean) @ u
        if s.max() - s.min() < min_len * 0.8:
            continue
        order = np.argsort(s)
        c = c[order][::3]
        out.append(c)
    return out


def straightness(pts, k1, k2=0.0):
    u = cv2.undistortPoints(pts.reshape(-1, 1, 2), K, np.array([k1, k2, 0, 0, 0]), P=K).reshape(-1, 2)
    m = u.mean(0)
    _, sv, vt = np.linalg.svd(u - m)
    d = np.abs((u - m) @ vt[1])
    return np.sqrt(np.mean(d ** 2)), d.max(), np.linalg.norm(u[-1] - u[0])


if __name__ == '__main__':
    ks = np.round(np.arange(-0.45, 0.151, 0.025), 3)
    total = np.zeros(len(ks))
    for cam in ('cam1', 'cam2'):
        img = cv2.imread('data/%s_bg.jpg' % cam)
        cs = chains(img)
        votes = []
        vis = (img * 0.4).astype(np.uint8)
        for c in cs:
            rms = np.array([straightness(c, k)[0] for k in ks])
            j = int(np.argmin(rms))
            span = straightness(c, ks[j])[2]
            r_edge = np.max(np.linalg.norm((c - [W / 2, H / 2]) / (W / 2), axis=1))
            # a chain is evidence only if it is really a line at its best k and bent at other ks
            if rms[j] < 1.2 and span > 350 and (rms.max() - rms[j]) > 1.0 and 0 < j < len(ks) - 1:
                votes.append((ks[j], span, r_edge, rms[j], rms[list(ks).index(0.0)]))
                w = span * r_edge ** 2
                total += w * np.minimum(rms, 6.0) / 6.0
                col = (0, 255, 0) if ks[j] < -0.05 else ((0, 0, 255) if ks[j] > 0.05 else (255, 255, 0))
                cv2.polylines(vis, [c.astype(np.int32)], False, col, 4)
        cv2.imwrite('data/%s_plumb.jpg' % cam, cv2.resize(vis, (1280, 720)))
        v = np.array(votes)
        print('%s: %d chains, %d straight-line votes' % (cam, len(cs), len(v)))
        if len(v):
            w = v[:, 1] * v[:, 2] ** 2
            print('   weighted median k1 = %.3f | k1 of chains reaching image edge (r>0.8): %s'
                  % (np.average(v[:, 0], weights=w), np.round(np.sort(v[v[:, 2] > 0.8, 0]), 3)))
            print('   rms at k1=0 vs at own best (px, edge chains):', np.round(v[v[:, 2] > 0.8][:, [4, 3]], 1).tolist()[:12])
    print('combined cost curve (lower = straighter), k1:', dict(zip(ks[::2], np.round(total[::2] / total.max(), 3))))
    print('combined best k1 = %.3f' % ks[int(np.argmin(total))])
