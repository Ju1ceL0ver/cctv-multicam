"""Rank already confirmed identities; distances are suggestions, not probabilities."""
from pathlib import Path
import numpy as np


def suggestions(folder, state, pieces, limit=3):
    d = Path(folder)
    path = d / 'emb_osnet_ain_x1_0_msmt17.npz'
    if not path.exists(): return []
    labels = state['labels']
    by_id = {p['piece']: p for p in state['pieces']}
    def feature(z, p):
        ids = p['dets'][::max(1, len(p['dets'])//32)]
        a = z[p['cam']][ids].astype(float)
        a = a / np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-8)
        f = np.median(a, axis=0)
        return f / max(np.linalg.norm(f), 1e-8)
    with np.load(path) as archive:
        z={cam:archive[cam] for cam in ('cam1','cam2')}
        targets = [feature(z, by_id[i]) for i in pieces if i in by_id]
        if not targets: return []
        scores = {}
        for i, p in by_id.items():
            label = labels.get(str(i))
            if not label or label == '?' or i in pieces: continue
            if state['quality'].get(str(i)) in ('false_positive', 'mixed', 'invalid'): continue
            if any(r['decision']=='different' and ((r['a'] in pieces and r['b']==i) or (r['b'] in pieces and r['a']==i)) for r in state['relations']): continue
            f = feature(z, p)
            distance = float(np.mean([1-t@f for t in targets]))
            if label not in scores or distance < scores[label]['distance']:
                scores[label] = {'label': label, 'piece': i, 'distance': round(distance, 4)}
    return sorted(scores.values(), key=lambda r:r['distance'])[:limit]
