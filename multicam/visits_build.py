"""Offline visit candidates from completed raw windows, with explicit coverage gaps.

Door anchors require an unambiguous time + floor match. Missing or ambiguous anchors
remain missing; presence in a 10-minute window is never called a complete visit.
usage: visits_build.py DAY [--no-media]
"""
import argparse
import json
from datetime import datetime, timedelta
from pathlib import Path
import numpy as np
from scipy.optimize import linear_sum_assignment
import visit_traj
from storage import atomic_json, read_json

ROOT = Path(__file__).resolve().parent
TAG = 'yolo26x-seg'


def clock(s):
    """Seconds on the day's local wall clock, independent of the host's timezone."""
    d = datetime.fromisoformat(s)
    return d.hour * 3600 + d.minute * 60 + d.second + d.microsecond / 1e6


def load_windows(day, root=ROOT):
    out, coverage = [], []
    for path in sorted((root / 'data/raw_clips').glob('*/people_%s.json' % TAG)):
        data = read_json(path, {})
        if not str(data.get('start', '')).replace('-', '').startswith(day):
            continue
        base = clock(data['start'])
        meta = read_json(path.parent / ('meta_%s.json' % TAG), {})
        coverage.append({'clip': path.parent.name, 'start': base, 'seconds': meta.get('seconds')})
        for p in data.get('people', []):
            t, xy = np.asarray(p['t'], float) + base, np.asarray(p['xy'], float)
            valid = np.isfinite(t) & np.isfinite(xy).all(axis=1)
            t, xy = t[valid], xy[valid]
            if len(t) < 3:
                continue
            order = np.argsort(t); t, xy = t[order], xy[order]
            sources = [dict(q, t=base + q['t'], raw_t=base + q.get('raw_t', q['t']),
                            clip=path.parent.name) for q in p.get('sources', [])]
            obs = np.asarray(p.get('observations', []),float)
            if obs.size:
                obs = obs[np.isfinite(obs).all(axis=1)]
                obs = obs[np.argsort(obs[:,0])]
                obs[:,0] += base
            out.append({'observations': obs, 'key': '%s:%s' % (path.parent.name, p['id']), 'clip': path.parent.name,
                        't': t, 'xy': xy, 'embedding': np.asarray(p.get('embedding', []), float),
                        'model': p.get('embedding_model'), 'sources': sources, 'height_m': p.get('height_m'),
                        'cams': p.get('cams', [])})
    return out, coverage


def join_windows(nodes, max_gap=120, appearance=0.25, margin=0.05):
    """Only mutual clear seam matches, with geometry and one identity per window.

    Existing thresholds from stitch.py; these visit joins still need owner validation.
    """
    candidates = []
    for i, a in enumerate(nodes):
        for j, b in enumerate(nodes):
            gap = b['t'][0] - a['t'][-1]
            if i == j or a['clip'] == b['clip'] or not -0.2 <= gap <= max_gap:
                continue
            ea, eb = a['embedding'], b['embedding']
            if not len(ea) or ea.shape != eb.shape or not a['model'] or a['model'] != b['model']:
                continue
            dist = np.linalg.norm(a['xy'][-1] - b['xy'][0])
            if dist > 0.5 + 2 * max(0, gap):
                continue
            cosine = 1 - float(ea @ eb / max(np.linalg.norm(ea) * np.linalg.norm(eb), 1e-9))
            cost = cosine + 0.0075 * max(0, gap - 30) / 60 + 0.02 * dist
            if cost <= appearance:
                candidates.append((cost, i, j))
    parent = list(range(len(nodes)))
    clips = {i: {a['clip']} for i, a in enumerate(nodes)}
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]; i = parent[i]
        return i
    links = []
    for cost, i, j in sorted(candidates):
        rivals = [v for v, a, b in candidates if (a == i or b == j) and (a, b) != (i, j)]
        if rivals and min(rivals) - cost < margin:
            continue
        a, b = find(i), find(j)
        if a == b or clips[a] & clips[b]:
            continue
        parent[b] = a; clips[a] |= clips.pop(b)
        links.append({'a': nodes[i]['key'], 'b': nodes[j]['key'], 'cost': round(cost, 4)})
    groups = {}
    for i in range(len(nodes)):
        groups.setdefault(find(i), []).append(nodes[i])
    return list(groups.values()), links


def door_events(day, root):
    from live_events import key_of
    updates = {}
    rp = root / 'data/live/records' / ('roles_%s.jsonl' % day)
    for p, target in [(rp, updates)]:
        if p.exists():
            for line in p.read_text(encoding='utf-8').splitlines():
                try:
                    r = json.loads(line); target[r['key']] = r
                except (ValueError, KeyError):
                    pass
    answers = read_json(root / 'data/visits' / day / 'people_answers.json', {})
    events = []
    path = root / 'data/live/events_cam1.jsonl'
    for line in path.read_text(encoding='utf-8').splitlines() if path.exists() else []:
        try:
            e = json.loads(line)
            if not e.get('clock', '').replace('-', '').startswith(day) or e.get('engine') != 'doorseg_c+zones':
                continue
            key = key_of(e); owner = answers.get('%.1f' % e['t'])
            if owner == 'x':
                continue
            role = updates.get(key, {}).get('role') or e.get('role')
            if owner is not None:
                role = 'staff' if owner in answers.get('_staff', []) else 'customer'
            events.append(dict(e, key=key, seconds=clock(e['clock']), role=role))
        except (ValueError, KeyError, TypeError):
            continue
    return sorted(events, key=lambda e: e['seconds'])


def anchor(groups, events, kind, door):
    """One event per group endpoint, one group per event, ambiguous cases left open."""
    a, b = np.asarray(door['door_a']), np.asarray(door['door_b'])
    ab = b-a
    selected = [e for e in events if e['kind'] == kind]
    costs = np.full((len(groups), len(selected)), 1e6)
    endpoint = 0 if kind == 'in' else -1
    for i, group in enumerate(groups):
        piece = min(group, key=lambda p: p['t'][0]) if kind == 'in' else max(group, key=lambda p: p['t'][-1])
        t = piece['t'][endpoint]
        # Restrict anchors to a visible endpoint by the physical doorway.
        point = piece['xy'][endpoint]
        projection = np.clip((point-a) @ ab / (ab @ ab), 0, 1)
        distance = float(np.linalg.norm(point-(a+projection*ab)))
        if distance > 1.5:
            continue
        for j, e in enumerate(selected):
            delta = abs(e['seconds'] - t)
            if delta <= 6:
                costs[i, j] = delta + distance
    result = {}
    if costs.size:
        rows, cols = linear_sum_assignment(costs)
        for i, j in zip(rows, cols):
            c = costs[i, j]
            rivals = np.r_[np.delete(costs[i], j), np.delete(costs[:, j], i)]
            if c < 1e6 and (not len(rivals) or np.min(rivals) - c >= 1):
                result[i] = {k: selected[j].get(k) for k in ('key', 'kind', 'clock', 'seconds', 'role', 'photo')}
    return result


def trajectory(group):
    times, points = [], []
    for node in sorted(group, key=lambda n: n['t'][0]):
        # Do not fabricate a smooth walk through recording gaps or long occlusions.
        obs = node.get('observations', np.empty((0,4)))
        ot = obs[:,0] if obs.size else node['t']
        oxy = obs[:,1:3] if obs.size else node['xy']
        foot = obs[:,3].astype(bool) if obs.size else np.zeros(len(ot),bool)
        for ids in np.split(np.arange(len(ot)), np.where(np.diff(ot) > 3)[0] + 1):
            if not len(ids):
                continue
            t, xy, speed, _ = visit_traj.smooth(np.round(ot[ids] / 0.2) * 0.2, oxy[ids], foot[ids])
            times.extend(t); points.extend(np.c_[xy, speed].tolist())
    order = np.argsort(times)
    return np.asarray(times)[order], np.asarray(points)[order]


def zone_times(t, points, zones):
    import cv2
    out = []
    for zone in zones:
        polygon = np.asarray(zone['polygon'], np.float32)
        inside = np.array([cv2.pointPolygonTest(polygon, tuple(map(float, xy[:2])), False) >= 0 for xy in points])
        dt = np.r_[np.minimum(np.diff(t), 0.2), 0.0]
        out.append({'name': zone['name'], 'seconds': round(float(dt[inside].sum()), 1)})
    return out


def build(day, root=ROOT, media=True):
    nodes, coverage = load_windows(day, root)
    if not nodes:
        raise ValueError('No completed windows for ' + day)
    groups, links = join_windows(nodes)
    events = door_events(day, root)
    door = read_json(root / 'data/plan_door.json', None)
    if door is None:
        raise ValueError('Missing floor door calibration')
    entries, exits = anchor(groups, events, 'in', door), anchor(groups, events, 'out', door)
    trajectories, records = {}, []
    zones = read_json(root / 'data/visits/floor_zones.json', [])
    folder = root / 'data/visits' / day; folder.mkdir(parents=True, exist_ok=True)
    for i, group in enumerate(groups):
        t, pos = trajectory(group)
        roles = {a['role'] for a in (entries.get(i), exits.get(i)) if a and a['role']}
        role = next(iter(roles)) if len(roles) == 1 else 'unknown'
        ident = 'v_' + min(n['key'] for n in group).replace(':', '_')
        trajectories[ident] = (t, pos[:, :2], pos[:, 2])
        record = {'id': ident, 'day': day, 'role': role, 'entry': entries.get(i), 'exit': exits.get(i),
                  'status': 'candidate', 'owner_verified': False,
                  'fragments': [n['key'] for n in group], 'height_m': None,
                  'start': float(t[0]), 'end': float(t[-1]),
                  'trajectory': np.c_[t, pos].round(3).tolist(), 'zones': zone_times(t, pos, zones),
                  'zones_status': 'configured' if zones else 'floor zones not configured', 'interactions': []}
        heights = [n['height_m'] for n in group if n['height_m'] is not None]
        if heights:
            record['height_m'] = round(float(np.median(heights)), 2)
        record['coverage_gaps'] = [[round(float(a), 2), round(float(b), 2)] for a, b in zip(t[:-1], t[1:]) if b-a > 3]
        record['door_bounded'] = bool(record['entry'] and record['exit'] and record['entry']['seconds'] < record['exit']['seconds'])
        record['media'] = {'status': 'not_requested' if not media else 'role_not_customer'}
        if media and role == 'customer':
            try:
                from visit_media import render
                record['media'] = render(day, ident, [s for n in group for s in n['sources']], folder)
            except Exception as exc:
                record['media'] = {'status': 'failed', 'error': str(exc)[:300]}
        records.append(record)
    staff = {r['id']: trajectories[r['id']] for r in records if r['role'] == 'staff'}
    for record in records:
        if record['role'] == 'customer':
            record['interactions'] = visit_traj.interactions(trajectories[record['id']], staff)
        atomic_json(folder / (record['id'] + '.json'), record)
    manifest = {'day': day, 'coverage': coverage, 'links': links, 'visits': records,
                'quality': {'verified_visits': 0, 'note': 'Candidate joins and interactions need owner labels'},
                'unanchored_events': [e['key'] for e in events if e['key'] not in {a['key'] for a in list(entries.values())+list(exits.values())}]}
    atomic_json(folder / 'index.json', manifest)
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('day'); parser.add_argument('--no-media', action='store_true')
    args = parser.parse_args()
    if not (len(args.day) == 8 and args.day.isdigit()):
        parser.error('DAY must be YYYYMMDD')
    result = build(args.day, media=not args.no_media)
    print('%s: %d candidates, %d window joins' % (args.day, len(result['visits']), len(result['links'])))
