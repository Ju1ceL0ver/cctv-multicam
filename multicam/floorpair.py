"""Pairs of floor points seen by both cameras, clicked by the owner on /floorpair -> the floor of camera 1 mapped
onto the floor of camera 2 (a homography on lens-undistorted pixels). Points are kept in raw 2560x1440 pixels.

data/floor_pairs.json {pairs: [{c1: [x, y], c2: [x, y]}], bg: name}; every save is appended to .history.jsonl."""
import json
import os
import time

import cv2
import numpy as np

RAW = (2560, 1440)


def _paths(root):
    d = os.path.join(root, 'data')
    return os.path.join(d, 'floor_pairs.json'), os.path.join(d, 'floor_pairs.history.jsonl'), \
        os.path.join(d, 'seg_datasets', 'backgrounds'), os.path.join(d, 'calib_final.json')


def backgrounds(root):
    """Empty-hall frames that exist for both cameras of the same file index: [(cam1 name, cam2 name)]."""
    b = _paths(root)[2]
    names = set(os.listdir(b)) if os.path.isdir(b) else set()
    by = {}
    for n in names:
        parts = n[:-4].split('_')
        if n.endswith('.jpg') and len(parts) == 4:
            by.setdefault((parts[0], parts[3]), {})[parts[1]] = n
    return [(v['cam1'], v['cam2']) for k, v in sorted(by.items()) if 'cam1' in v and 'cam2' in v]


def undistort(root, cam, pts):
    calib = _paths(root)[3]
    pts = np.asarray(pts, np.float64).reshape(-1, 1, 2)
    if not os.path.exists(calib) or not len(pts):
        return pts.reshape(-1, 2)
    p = json.load(open(calib))[cam]
    K = np.array(p['K'])
    return cv2.undistortPoints(pts, K, np.array(p['dist']), P=K).reshape(-1, 2)


def _fit(a, b):
    H, _ = cv2.findHomography(a, b, 0)
    if H is None:
        return None, None
    pb = cv2.perspectiveTransform(a.reshape(-1, 1, 2), H).reshape(-1, 2)
    pa = cv2.perspectiveTransform(b.reshape(-1, 1, 2), np.linalg.inv(H)).reshape(-1, 2)
    return H, (np.linalg.norm(pb - b, axis=1) + np.linalg.norm(pa - a, axis=1)) / 2


def fit(root, pairs):
    """Homography cam1 -> cam2 on raw pixels and on lens-undistorted ones; the one with the smaller median
    error is kept (the stored lens model can do more harm than good here). Each pair's error in pixels."""
    if len(pairs) < 4:
        return None
    a = np.array([p['c1'] for p in pairs], np.float64)
    b = np.array([p['c2'] for p in pairs], np.float64)
    best = None
    for mode, (pa, pb) in (('raw', (a, b)), ('undistorted', (undistort(root, 'cam1', a), undistort(root, 'cam2', b)))):
        H, err = _fit(pa, pb)
        if H is not None and (best is None or np.median(err) < np.median(best[2])):
            best = (mode, H, err)
    if best is None:
        return None
    mode, H, err = best
    return {'H': H.tolist(), 'mode': mode, 'err': [round(float(e), 1) for e in err],
            'median': round(float(np.median(err)), 1), 'max': round(float(err.max()), 1)}


def project(root, H, cam, pts, mode='undistorted'):
    """Raw pixels of `cam` -> raw pixels of the other camera (undistort, homography, distort back; in 'raw'
    mode the homography alone)."""
    H = np.asarray(H, np.float64)
    other = 'cam2' if cam == 'cam1' else 'cam1'
    if mode == 'raw':
        p = np.asarray(pts, np.float64).reshape(-1, 1, 2)
        return cv2.perspectiveTransform(p, H if cam == 'cam1' else np.linalg.inv(H)).reshape(-1, 2)
    u = undistort(root, cam, pts).reshape(-1, 1, 2)
    v = cv2.perspectiveTransform(u, H if cam == 'cam1' else np.linalg.inv(H)).reshape(-1, 2)
    calib = _paths(root)[3]
    if not os.path.exists(calib):
        return v
    p = json.load(open(calib))[other]
    K, dist = np.array(p['K']), np.array(p['dist'])
    n = cv2.undistortPoints(v.reshape(-1, 1, 2), K, np.zeros(5)).reshape(-1, 2)     # to normalised coordinates
    out, _ = cv2.projectPoints(np.c_[n, np.ones(len(n))], np.zeros(3), np.zeros(3), K, dist)
    return out.reshape(-1, 2)


def register(app, root):
    from flask import jsonify, request, render_template, send_file, abort, Response

    @app.get('/floorpair')
    def floorpair_page():
        return render_template('floorpair.html')

    @app.get('/api/floorpair')
    def floorpair_get():
        path = _paths(root())[0]
        st = json.load(open(path)) if os.path.exists(path) else {'pairs': []}
        return jsonify({'state': st, 'fit': fit(root(), st['pairs']), 'backgrounds': backgrounds(root())})

    @app.post('/api/floorpair')
    def floorpair_save():
        path, hist, _, _ = _paths(root())
        body = request.get_json(force=True)
        pairs = [{'c1': [float(p['c1'][0]), float(p['c1'][1])], 'c2': [float(p['c2'][0]), float(p['c2'][1])]}
                 for p in body.get('pairs', [])]
        st = {'pairs': pairs, 'bg': body.get('bg'), 'saved': time.strftime('%Y-%m-%d %H:%M:%S')}
        if body.get('manual'):                    # the owner's hand correction on top of the fit, and the result
            st['manual'] = body['manual']
            st['H_final'] = body.get('H_final')
        tmp = path + '.tmp'
        json.dump(st, open(tmp, 'w'), indent=1)
        os.replace(tmp, path)
        with open(hist, 'a') as f:
            f.write(json.dumps(st) + '\n')
        return jsonify({'ok': True, 'fit': fit(root(), pairs)})

    @app.post('/api/floorpair/project')
    def floorpair_project():
        body = request.get_json(force=True)
        return jsonify({'pts': project(root(), body['H'], body['cam'], body['pts'], body.get('mode', 'undistorted')).tolist()})

    @app.get('/api/floorpair/warp')
    def floorpair_warp():
        """Camera 1's frame drawn in camera 2's view by H (raw pixels), with alpha outside the warped area."""
        name = os.path.basename(request.args['bg'])
        img = cv2.imread(os.path.join(_paths(root())[2], name))
        if img is None:
            abort(404)
        H = np.array(json.loads(request.args['h']), np.float64).reshape(3, 3)
        k = img.shape[1] / RAW[0]
        S = np.diag([k, k, 1.0])
        Hi = S @ H @ np.linalg.inv(S)
        size = (img.shape[1], img.shape[0])
        warped = cv2.warpPerspective(img, Hi, size)
        alpha = cv2.warpPerspective(np.full(img.shape[:2], 255, np.uint8), Hi, size)
        ok, buf = cv2.imencode('.webp', np.dstack([warped, alpha]), [cv2.IMWRITE_WEBP_QUALITY, 80])
        return Response(buf.tobytes(), mimetype='image/webp')

    @app.get('/api/floorpair/bg/<name>')
    def floorpair_bg(name):
        p = os.path.join(_paths(root())[2], os.path.basename(name))
        if not os.path.exists(p):
            abort(404)
        return send_file(p, mimetype='image/jpeg', max_age=3600)
