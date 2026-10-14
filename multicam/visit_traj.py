"""Smooth floor trajectory of one person and staff interactions (owner's plan, AGENTS.md section 8).

smooth(): constant-velocity Kalman on the floor (metres) with per-observation noise (foot seen vs
placed by head and height), outlier rejection by Mahalanobis gate, a speed cap and an RTS pass back
in time -- a visit is written only after the person leaves, so the delay costs nothing.
interactions(): customer and staff closer than ~1.2 m for longer than ~10 s, both nearly standing."""
import numpy as np

SIG_FOOT, SIG_HEAD = 0.15, 0.40       # metres, observation noise
ACC = 1.5                             # m/s^2, process noise (walking start/stop)
VMAX = 2.5                            # m/s
GATE = 3.5                            # Mahalanobis, sigma units
MAX_STREAK = 5                        # rejections in a row before the filter re-anchors


def smooth(t, xy, foot=None, step=0.2):
    """t (s), xy (n,2) metres, foot (n,) bool. Returns grid times, smoothed xy, speed, kept-mask of inputs."""
    t = np.asarray(t, float); xy = np.asarray(xy, float)
    o = np.argsort(t, kind='stable'); t, xy = t[o], xy[o]
    foot = np.ones(len(t), bool) if foot is None else np.asarray(foot, bool)[o]
    ok = np.all(np.isfinite(xy), 1)
    t, xy, foot = t[ok], xy[ok], foot[ok]
    if len(t) == 0:
        return np.zeros(0), np.zeros((0, 2)), np.zeros(0), ok
    grid = np.arange(t[0], t[-1] + 1e-9, step)
    # observations binned onto the grid, weighted mean per bin
    b = np.clip(np.round((t - t[0]) / step).astype(int), 0, len(grid) - 1)
    w = np.where(foot, 1 / SIG_FOOT ** 2, 1 / SIG_HEAD ** 2)
    W = np.bincount(b, w, len(grid))
    Z = np.c_[np.bincount(b, w * xy[:, 0], len(grid)), np.bincount(b, w * xy[:, 1], len(grid))]
    has = W > 0
    Z[has] /= W[has, None]
    Rv = np.where(has, 1 / np.maximum(W, 1e-9), np.inf)
    F = np.eye(4); F[0, 2] = F[1, 3] = step
    q = ACC ** 2
    Q = q * np.array([[step ** 4 / 4, 0, step ** 3 / 2, 0], [0, step ** 4 / 4, 0, step ** 3 / 2],
                      [step ** 3 / 2, 0, step ** 2, 0], [0, step ** 3 / 2, 0, step ** 2]])
    H = np.eye(2, 4)
    n = len(grid)
    xs, Ps, xp, Pp = np.zeros((n, 4)), np.zeros((n, 4, 4)), np.zeros((n, 4)), np.zeros((n, 4, 4))
    first = np.argmax(has)
    x = np.r_[Z[first], 0, 0]; P = np.diag([SIG_HEAD ** 2] * 2 + [1.0, 1.0])
    rejected = np.zeros(n, bool)
    streak = 0
    for k in range(n):
        if k:
            x, P = F @ x, F @ P @ F.T + Q
        xp[k], Pp[k] = x, P
        if has[k]:
            S = H @ P @ H.T + np.eye(2) * Rv[k]
            r = Z[k] - H @ x
            d2 = r @ np.linalg.solve(S, r)
            if d2 <= GATE ** 2 or k == first or streak >= MAX_STREAK:
                if streak >= MAX_STREAK:      # the person really moved: re-anchor
                    P = P + np.diag([d2 * Rv[k]] * 2 + [1.0, 1.0])
                    S = H @ P @ H.T + np.eye(2) * Rv[k]
                streak = 0
                K = P @ H.T @ np.linalg.inv(S)
                x = x + K @ r; P = (np.eye(4) - K @ H) @ P
            else:
                rejected[k] = True; streak += 1
        v = np.hypot(x[2], x[3])
        if v > VMAX:
            x[2:] *= VMAX / v
        xs[k], Ps[k] = x, P
    for k in range(n - 2, -1, -1):        # RTS backward pass
        C = Ps[k] @ F.T @ np.linalg.inv(Pp[k + 1])
        xs[k] = xs[k] + C @ (xs[k + 1] - xp[k + 1])
        Ps[k] = Ps[k] + C @ (Ps[k + 1] - Pp[k + 1]) @ C.T
    keep = np.ones(len(ok), bool)
    idx = np.nonzero(ok)[0]
    keep[o[idx]] = ~rejected[b]
    return grid, xs[:, :2], np.hypot(xs[:, 2], xs[:, 3]), keep


def interactions(cust, staff, near=1.2, min_s=10.0, still=0.4, gap_s=3.0):
    """cust: (grid, xy, speed); staff: {id: (grid, xy, speed)}. Same step on both grids.
    Returns [{staff, start, end, seconds, x, y}]."""
    g, xy, sp = cust
    out = []
    for sid, (gs, xys, sps) in staff.items():
        common, ia, ib = np.intersect1d(np.round(g, 1), np.round(gs, 1), return_indices=True)
        if not len(common):
            continue
        d = np.linalg.norm(xy[ia] - xys[ib], axis=1)
        on = (d < near) & (sp[ia] < still) & (sps[ib] < still)
        # runs of "on", short holes (occlusion, a step aside) bridged
        runs, s0, last = [], None, None
        for k in np.nonzero(on)[0]:
            if s0 is not None and common[k] - common[last] > gap_s:
                runs.append((s0, last)); s0 = None
            if s0 is None:
                s0 = k
            last = k
        if s0 is not None:
            runs.append((s0, last))
        for s, e in runs:
            dur = common[e] - common[s]
            if dur >= min_s:
                m = (xy[ia][s:e + 1] + xys[ib][s:e + 1]) / 2
                out.append({'staff': sid, 'start': float(common[s]), 'end': float(common[e]),
                            'seconds': round(float(dur), 1), 'x': round(float(m[:, 0].mean()), 2),
                            'y': round(float(m[:, 1].mean()), 2)})
    return sorted(out, key=lambda r: r['start'])


def roughness(grid, xy):
    """Jitter measures: p90 and max step speed (m/s), share of steps above VMAX."""
    if len(grid) < 3:
        return {}
    v = np.linalg.norm(np.diff(xy, axis=0), axis=1) / np.maximum(np.diff(grid), 1e-3)
    return {'v_p90': round(float(np.percentile(v, 90)), 2), 'v_max': round(float(v.max()), 2),
            'over_vmax': round(float((v > VMAX).mean()), 3)}
