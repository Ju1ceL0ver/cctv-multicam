"""SAM 3.1's chunks (sam31_video.py) -> people: link the chunks at their shared frames, join the pieces of one person by the heavy ReID teachers, draw the video.

1. Seams. Chunk c+1 starts OVERLAP frames before chunk c ends. For every pair (track of c, track of c+1) the
   mean mask IoU over the shared frames where either is seen; Hungarian on that, a link when it is >= SEAM_IOU.
   The shared frames keep chunk c's masks. Unlinked pieces stay separate -- ReID gets its chance at them.
2. ReID. Per piece up to K views (the largest, surest, uncovered, not at the frame's edge, spread in time),
   crops (box + 5 %) from the full frame; TransReID (clothes) + CSCI (body shape), track_emb.teachers.
   Distance = 1 - (cos clothes + cos shape) / 2 of the pieces' mean vectors. Greedy, cheapest pair first,
   while below TAU and the two groups never share a frame (two pieces seen together are two people).
Everything SAM calls a person stays a person here: posters, strollers and scraps are the owner's small filter
network's job, afterwards.

usage (venv_rfdetr): sam31_reid.py TAG [TAU] [K]
  -> data/logs/sam31/TAG/report.json, sheet.jpg (every piece's views), people.mp4"""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
SEAM_IOU = 0.3
FPS = 12.5
COL = [(66, 197, 245), (80, 220, 100), (245, 160, 66), (220, 90, 220), (240, 220, 70), (160, 100, 255), (60, 200, 200),
       (200, 160, 90), (120, 230, 180), (250, 120, 160), (90, 140, 250), (230, 230, 230)]


class Masks:
    def __init__(self, path, mmap=False):
        """mmap: read the packed masks from chunks_buf.npy (written once by to_npy) mapped, not loaded -- the training's
        data workers then share one copy in the OS cache instead of each holding the window's masks (0.2-0.6 GB)."""
        path = Path(path)
        npy = path.with_name('chunks_buf.npy')
        if mmap and npy.exists() and npy.stat().st_mtime >= path.stat().st_mtime:
            with np.load(path) as z:
                self.rows, self.offs = z['rows'], z['offs']
            self.buf = np.load(npy, mmap_mode='r')
        else:
            with np.load(path) as z:
                self.rows, self.buf, self.offs = z['rows'], z['buf'], z['offs']

    @staticmethod
    def to_npy(path):
        """chunks.npz -> chunks_buf.npy next to it (uncompressed, for mmap); nothing when it is already there."""
        path = Path(path)
        npy = path.with_name('chunks_buf.npy')
        if npy.exists() and npy.stat().st_mtime >= path.stat().st_mtime:
            return npy
        with np.load(path) as z:
            buf = z['buf']
        tmp = npy.with_name('chunks_buf.tmp.npy')
        np.save(tmp, buf)
        tmp.replace(npy)
        return npy

    def crop(self, r):
        """The mask of row r inside its box."""
        x1, y1, x2, y2 = self.rows[r, 4:8].astype(int)
        h, w = y2 - y1, x2 - x1
        return np.unpackbits(np.asarray(self.buf[self.offs[r]:self.offs[r + 1]]))[:h * w].reshape(h, w).astype(bool)

    def iou(self, a, b):
        A, B = self.rows[a, 4:8], self.rows[b, 4:8]
        X1, Y1, X2, Y2 = max(A[0], B[0]), max(A[1], B[1]), min(A[2], B[2]), min(A[3], B[3])
        if X2 <= X1 or Y2 <= Y1:
            return 0.0
        ma, mb = self.crop(a), self.crop(b)
        sa = ma[int(Y1 - A[1]):int(Y2 - A[1]), int(X1 - A[0]):int(X2 - A[0])]
        sb = mb[int(Y1 - B[1]):int(Y2 - B[1]), int(X1 - B[0]):int(X2 - B[0])]
        inter = float((sa & sb).sum())
        return inter / max(1.0, ma.sum() + mb.sum() - inter)


def link_seams(M, overlap):
    ov = (lambda s: overlap.get(s, 0)) if isinstance(overlap, dict) else (lambda s: overlap)   # per session or one for all
    """{(chunk, local id): piece id}, the rows each piece owns, and the seams' diagnostics."""
    from scipy.optimize import linear_sum_assignment
    rows = M.rows
    starts = sorted({int(s) for s in rows[:, 0]})
    piece, nxt, seams = {}, 0, []
    for c, s in enumerate(starts):
        here = sorted({int(i) for i in rows[rows[:, 0] == s, 2]})
        mapping = {}
        if c:
            prev = starts[c - 1]
            shared = range(s, s + ov(s))
            A = {}      # (prev id, cur id) -> summed IoU; present -> frames where either is seen
            seen = {}
            for f in shared:
                pr = [r for r in np.nonzero((rows[:, 0] == prev) & (rows[:, 1] == f))[0]]
                cr = [r for r in np.nonzero((rows[:, 0] == s) & (rows[:, 1] == f))[0]]
                for a in pr:
                    for b in cr:
                        key = (int(rows[a, 2]), int(rows[b, 2]))
                        seen[key] = seen.get(key, 0)
                        v = M.iou(a, b)
                        if v > 0:
                            A[key] = A.get(key, 0.0) + v
                        seen[key] += 1
            pids = sorted({k[0] for k in seen}); cids = sorted({k[1] for k in seen})
            if pids and cids:
                S = np.array([[A.get((p, q), 0.0) / max(1, seen.get((p, q), 1)) for q in cids] for p in pids])
                r, cc = linear_sum_assignment(-S)
                for i, j in zip(r, cc):
                    if S[i, j] >= SEAM_IOU:
                        mapping[cids[j]] = piece[(prev, pids[i])]
                seams.append({'at': s, 'prev': len(pids), 'cur': len(cids), 'linked': len(mapping),
                              'best_unlinked': sorted([round(float(S[i, j]), 2) for i, j in zip(r, cc) if S[i, j] < SEAM_IOU], reverse=True)[:5]})
        for i in here:
            if i not in mapping:
                mapping[i] = nxt; nxt += 1
            piece[(s, i)] = mapping[i]
    owned = {}
    for r, (s, f, i) in enumerate(rows[:, :3].astype(int)):
        if s == starts[0] or f >= s + ov(s):            # the shared frames keep the previous chunk's masks
            owned.setdefault(piece[(s, i)], []).append(r)
    return owned, seams


def main_blob(M, r):
    """(box of the largest connected piece of row r's mask in frame pixels, its share of the mask's area).
    SAM's mask of one person is sometimes several pieces: the person cut by a pole, or a stray patch far away
    (on the gallery poster); a box around all of them is no crop of the person."""
    import cv2
    m = M.crop(r).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(m, 8)
    if n < 2:
        return tuple(M.rows[r, 4:8]), 0.0
    j = 1 + int(np.argmax(st[1:, 4]))
    x, y, w, h, a = st[j]
    x0, y0 = M.rows[r, 4], M.rows[r, 5]
    return (x0 + x, y0 + y, x0 + x + w, y0 + y + h), float(a) / max(1.0, float(st[1:, 4].sum()))


def views(M, owned, k, W, H):
    """{piece: [(row, box)]} -- up to k clean views of every piece: the largest, surest, whole (its mask one
    piece), uncovered by another mask's box, not at the frame's edge, spread in time; box = the main piece's."""
    rows = M.rows
    by_frame = {}
    for p, rs in owned.items():
        for r in rs:
            by_frame.setdefault(int(rows[r, 1]), []).append(r)
    out = {}
    for p, rs in owned.items():
        rs = sorted(rs, key=lambda r: rows[r, 1])
        pick = []
        for chunk in np.array_split(np.array(rs), min(k, len(rs))):
            best, bs = None, -1.0
            for r in chunk:
                box, share = main_blob(M, int(r))
                x1, y1, x2, y2 = box
                if best is None:
                    best = (int(r), box)                   # the first when every score is NaN
                cover = 0.0
                for q in by_frame[int(rows[r, 1])]:
                    if q == r:
                        continue
                    ix = max(0.0, min(x2, rows[q, 6]) - max(x1, rows[q, 4])); iy = max(0.0, min(y2, rows[q, 7]) - max(y1, rows[q, 5]))
                    cover = max(cover, ix * iy / max(1.0, (x2 - x1) * (y2 - y1)))
                edge = x1 <= 2 or y1 <= 2 or x2 >= W - 2 or y2 >= H - 2
                s = (y2 - y1) * (x2 - x1) * rows[r, 3] * (0.1 if cover >= 0.3 else 1.0) * (0.3 if edge else 1.0) * (0.1 if share < 0.9 else 1.0)
                if s > bs:
                    best, bs = (int(r), box), s
            pick.append(best)
        out[p] = pick
    return out


def join(ids, D, frames, tau):
    """Greedy join with cannot-link on shared frames -> {piece: group root}."""
    parent = {i: i for i in ids}
    fr = {i: set(frames[i]) for i in ids}

    def find(x):
        while parent[x] != x:
            x = parent[x]
        return x
    pairs = sorted((D[a][b], ia, ib) for a, ia in enumerate(ids) for b, ib in enumerate(ids) if a < b)
    for d, a, b in pairs:
        if d > tau:
            break
        ra, rb = find(a), find(b)
        if ra == rb or fr[ra] & fr[rb]:
            continue
        parent[rb] = ra
        fr[ra] |= fr.pop(rb)
    return {i: find(i) for i in ids}


def main():
    import cv2
    import track_emb
    import day_proxy
    import os
    tag = sys.argv[1]
    tau = float(sys.argv[2]) if len(sys.argv) > 2 else 0.35
    k = int(sys.argv[3]) if len(sys.argv) > 3 else 8
    base = (Path(os.environ['RA_S31_ROOT']) if os.environ.get('RA_S31_ROOT') else ROOT / 'data' / 'logs' / 'sam31') / tag
    video = os.environ.get('RA_S31_VIDEO', '1') == '1'
    info = json.load(open(base / 'info.json'))
    W, H = info['size']
    M = Masks(base / 'chunks.npz')
    rows = M.rows
    overlap = {int(s): int(sh) for s, e, sh in info['sessions']} if 'sessions' in info else info['overlap']
    owned, seams = link_seams(M, overlap)
    frames_of = {p: sorted({int(rows[r, 1]) for r in rs}) for p, rs in owned.items()}
    pick = views(M, owned, k, W, H)
    if (base / 'frames').exists():
        frame = lambda f: cv2.imread(str(base / 'frames' / ('%05d.jpg' % int(f))))
    else:                                        # a whole stretch: the frames are in video.mp4, read forward once
        need = sorted({int(rows[r, 1]) for p in pick for r, _ in pick[p]})
        got, cap, k = {}, cv2.VideoCapture(str(base / 'video.mp4')), -1
        for f in need:
            while k < f and cap.grab():
                k += 1
            if k == f:
                got[f] = cap.retrieve()[1]
        cap.release()
        frame = lambda f: got.get(int(f))
    cache = {}
    crops, owner = [], []
    for p in sorted(pick):
        for r, (x1, y1, x2, y2) in pick[p]:
            w, h = x2 - x1, y2 - y1
            f = int(rows[r, 1])
            img = cache.setdefault(f, frame(f))
            c = img[max(0, int(y1 - 0.05 * h)):min(H, int(y2 + 0.05 * h)), max(0, int(x1 - 0.05 * w)):min(W, int(x2 + 0.05 * w))]
            crops.append(c.copy()); owner.append(p)
    cache.clear()
    embed = track_emb.teachers()
    v1, v2 = embed(crops)
    owner = np.array(owner)
    ids = sorted(owned)
    norm = lambda m: m / np.linalg.norm(m)
    C = np.stack([norm(v1[owner == p].mean(0)) for p in ids]) if ids else np.zeros((0, 1))
    S = np.stack([norm(v2[owner == p].mean(0)) for p in ids]) if ids else np.zeros((0, 1))
    D = 1.0 - (C @ C.T + S @ S.T) / 2.0
    rep = {'info': info, 'tau': tau, 'sam_tracks': int(len({(int(a), int(b)) for a, b in rows[:, [0, 2]]})),
           'pieces': len(owned), 'seams': seams,
           'piece': {str(p): {'frames': [frames_of[p][0], frames_of[p][-1]], 'n': len(frames_of[p])} for p in owned},
           'distance': {'%d-%d' % (a, b): round(float(D[i][j]), 3) for i, a in enumerate(ids) for j, b in enumerate(ids) if i < j},
           'groups': {}}
    for t in (0.2, 0.25, 0.3, 0.35, 0.4, 0.45):
        g = join(ids, D, frames_of, t)
        groups = {}
        for i in ids:
            groups.setdefault(g[i], []).append(i)
        rep['groups'][str(t)] = {'people': len(groups), 'joined': [v for v in groups.values() if len(v) > 1]}
    g = join(ids, D, frames_of, tau)
    roots = sorted(set(g.values()), key=lambda r: min(frames_of[i][0] for i in ids if g[i] == r))
    number = {r: n + 1 for n, r in enumerate(roots)}
    person = {p: number[g[p]] for p in ids}
    rep['people'] = len(roots)
    rep['person_of_piece'] = {str(p): person.get(p) for p in owned}
    json.dump(rep, open(base / 'report.json', 'w'), indent=1)
    # the sheet: one row per piece
    strips = []
    for p in sorted(owned):
        cs = [cv2.resize(c, (max(1, int(c.shape[1] * 200 / max(1, c.shape[0]))), 200)) for c, o in zip(crops, owner) if o == p]
        lab = np.zeros((200, 260, 3), np.uint8)
        who = 'P%d' % person[p]
        col = COL[person[p] % len(COL)]
        cv2.putText(lab, 'piece %d' % p, (8, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        cv2.putText(lab, who, (8, 110), cv2.FONT_HERSHEY_SIMPLEX, 1.8, col, 4)
        cv2.putText(lab, 'f %d-%d' % (frames_of[p][0], frames_of[p][-1]), (8, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        strips.append(np.hstack([lab] + cs))
    wmax = max(s.shape[1] for s in strips)
    sheet = np.vstack([np.hstack([s, np.zeros((200, wmax - s.shape[1], 3), np.uint8)]) for s in strips])
    cv2.imwrite(str(base / 'sheet.jpg'), sheet, [cv2.IMWRITE_JPEG_QUALITY, 85])
    if not video:
        print(json.dumps({'tag': tag, 'sam_tracks': rep['sam_tracks'], 'pieces': len(owned), 'people': len(roots)}))
        return
    # the video
    per = {}
    for p, rs in owned.items():
        for r in rs:
            per.setdefault(int(rows[r, 1]), []).append((r, p))
    raw_out = base / 'people_raw.mp4'
    vo = cv2.VideoWriter(str(raw_out), cv2.VideoWriter_fourcc(*'mp4v'), FPS, (W, H))
    for f in range(info['frames']):
        img = frame(f)
        for r, p in per.get(f, []):
            x1, y1, x2, y2 = rows[r, 4:8].astype(int)
            m = np.zeros((H, W), bool); m[y1:y2, x1:x2] = M.crop(r)
            col = COL[person[p] % len(COL)]
            over = img.copy(); over[m] = col
            img = cv2.addWeighted(img, 0.72, over, 0.28, 0)
            cs, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(img, cs, -1, (0, 0, 0), 7); cv2.drawContours(img, cs, -1, col, 3)
            txt = 'P%d' % person[p]
            x, y = (x1 + x2) // 2, max(30, y1 - 10)
            cv2.putText(img, txt, (x - 50, y), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 8)
            cv2.putText(img, txt, (x - 50, y), cv2.FONT_HERSHEY_SIMPLEX, 1.5, col, 3)
        cv2.rectangle(img, (0, 0), (W, 60), (0, 0, 0), -1)
        cv2.putText(img, 'SAM 3.1 + seams + ReID tau %.2f | %s | frame %d | %d SAM tracks -> %d pieces -> %d people'
                    % (tau, tag, f, rep['sam_tracks'], len(owned), len(roots)), (16, 44), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
        vo.write(img)
    vo.release()
    subprocess.run([day_proxy._ffmpeg(), '-y', '-loglevel', 'error', '-i', str(raw_out), '-c:v', 'libx264', '-preset', 'veryfast',
                    '-crf', '24', '-pix_fmt', 'yuv420p', str(base / 'people.mp4')])
    raw_out.unlink()
    print(json.dumps({'tag': tag, 'sam_tracks': rep['sam_tracks'], 'pieces': len(owned), 'people': len(roots), 'seams': seams, 'groups': rep['groups']}))


if __name__ == '__main__':
    main()
