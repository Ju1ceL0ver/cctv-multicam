"""Atomic, versioned human decisions and exact-frame video corrections.

Machine suggestions never become human approvals by being saved. Legacy files are
compatibility views; review_state.json remains the source of truth.
"""
import copy
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from storage import atomic_json, file_lock, read_json

TAG = 'yolo26x-seg'
MASK_STATES = {'unreviewed', 'valid', 'invalid', 'false_positive', 'mixed'}


class Conflict(ValueError):
    pass


def read_state(folder):
    d = Path(folder)
    state = read_json(d / 'review_state.json')
    if state is not None:
        return state
    pieces = read_json(d / ('pieces_' + TAG + '.json'), [])
    labels = read_json(d / 'gt_manual.json', {})
    return {'revision': 0, 'pieces': pieces, 'labels': labels, 'quality': {},
            'relations': [], 'seconds_reviewing': 0, 'last_operation': None}


def fingerprint(state):
    evidence = {k: state.get(k) for k in ('pieces', 'labels', 'quality', 'relations', 'video')}
    return hashlib.sha256(json.dumps(evidence, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def _identity_view(d, state):
    if state.get('video'):
        from video_annotations import identity_materialization
        return identity_materialization(d, state)
    gt = {'cam1': {}, 'cam2': {}}
    for p in state['pieces']:
        label = state['labels'].get(str(p['piece']))
        if label and label != '?' and state['quality'].get(str(p['piece'])) not in ('false_positive', 'mixed'):
            gt[p['cam']].update({str(i): label for i in p['dets']})
    return gt


def materialize(d, state, identity=None):
    """Compatibility views; indices in immutable teacher arrays never change."""
    d = Path(d)
    atomic_json(d / 'gt_manual.json', state['labels'])
    atomic_json(d / ('gt_identity_' + TAG + '.json'), _identity_view(d, state) if identity is None else identity)


def _apply(d, state, operation):
    from video_annotations import VIDEO_ACTIONS, apply_operation, clear_intervals, _label
    action = operation.get('action', 'label')
    by_id = {int(p['piece']): p for p in state['pieces']}
    ids = [int(i) for i in operation.get('pieces', [])]
    if any(i not in by_id for i in ids):
        raise ValueError('Неизвестный отрезок')
    if action in VIDEO_ACTIONS:
        apply_operation(d, state, operation)
    elif action in ('label', 'quality'):
        if not ids:
            raise ValueError('Выберите отрезки')
        if action == 'label':
            label = _label(state, operation.get('label'))
            clear_intervals(state, 'label', ids)
            for i in ids:
                if label:
                    state['labels'][str(i)] = label
                else:
                    state['labels'].pop(str(i), None)
        else:
            quality = operation.get('quality')
            if quality not in MASK_STATES:
                raise ValueError('Неизвестное состояние маски')
            clear_intervals(state, 'quality', ids)
            for i in ids:
                state['quality'][str(i)] = quality
    elif action == 'merge':
        src, dst = operation['from'], operation['to']
        if src == '?' or dst == '?' or not dst:
            raise ValueError('Неизвестные личности нельзя объединять')
        _label(state, src); _label(state, dst)
        state['labels'] = {k: dst if v == src else v for k, v in state['labels'].items()}
        video = state.get('video', {})
        for interval in video.get('intervals', []):
            if interval['kind'] == 'label' and interval['value'] == src:
                interval['value'] = dst
        for obs in video.get('observations', {}).values():
            if obs.get('label') == src:
                obs['label'] = dst
    elif action == 'split':
        import numpy as np
        i = int(operation['piece'])
        if i not in by_id:
            raise ValueError('Неизвестный отрезок')
        p = by_id[i]
        with np.load(d / ('dets_' + TAG + '.npz')) as z:
            times = z[p['cam']][p['dets'], 0]
        at = float(operation['at'])
        before = [int(di) for di, t in zip(p['dets'], times) if t < at]
        after = [int(di) for di, t in zip(p['dets'], times) if t >= at]
        if not before or not after:
            raise ValueError('Выберите момент внутри отрезка')
        child = copy.deepcopy(p)
        child['piece'] = max(by_id) + 1
        for item, dets in ((p, before), (child, after)):
            item['origin_piece'] = p.get('origin_piece', i)
            item['dets'] = dets; item['n_dets'] = len(dets)
            wanted = set(dets)
            selected = [j for j, di in enumerate(before + after) if di in wanted]
            raw0, raw1 = float(times[selected[0]]), float(times[selected[-1]])
            item['raw_t0'], item['raw_t1'] = raw0, raw1
            item['t0'], item['t1'] = raw0, raw1
            item['review_split'] = True
        state['relations'] = [r for r in state['relations'] if i not in (r['a'], r['b'])]
        state['pieces'].append(child)
        clear_intervals(state, 'label', [i])
        clear_intervals(state, 'quality', [i])
        for oid, obs in state.get('video', {}).get('observations', {}).items():
            if obs.get('piece') == i:
                if obs.get('frame', 0) >= at * 25:
                    obs['piece'] = child['piece']
                obs.pop('label', None)
                obs['quality'] = 'unreviewed'
                obs['mask_review_source'] = 'unreviewed'
        state['labels'].pop(str(i), None)
        state['labels'].pop(str(child['piece']), None)
        state['quality'].pop(str(i), None)
    elif action == 'relation':
        a, b = int(operation['a']), int(operation['b'])
        decision = operation['decision']
        if a == b or a not in by_id or b not in by_id or decision not in ('same', 'different', 'unsure'):
            raise ValueError('Неверная пара')
        a, b = sorted((a, b))
        state['relations'] = [r for r in state['relations'] if (r['a'], r['b']) != (a, b)]
        state['relations'].append({'a': a, 'b': b, 'decision': decision})
        la, lb = state['labels'].get(str(a)), state['labels'].get(str(b))
        if decision == 'same':
            known = next((v for v in (la, lb) if v and v != '?'), None) or _label(state, 'NEW')
            clear_intervals(state, 'label', [a, b])
            state['labels'][str(a)] = known
            state['labels'][str(b)] = known
        elif decision == 'different' and la and la == lb and la != '?':
            state['labels'].pop(str(b), None)
            clear_intervals(state, 'label', [b])
    else:
        raise ValueError('Неизвестное действие')


def transact(folder, operation, expected=None):
    from video_annotations import validate_relations, _label
    d = Path(folder)
    with file_lock(d / '.review.lock'):
        old = read_state(d)
        if expected is not None and int(expected) != old['revision']:
            raise Conflict('Разметка изменилась в другой вкладке. Обновите страницу.')
        state = copy.deepcopy(old)
        action = operation.get('action', 'label')
        if action == 'undo':
            rev = old.get('undo_revision')
            if rev is None:
                raise ValueError('Нет действия для отмены')
            state = read_json(d / 'review_history' / ('%08d.json' % rev))
            if state is None:
                raise ValueError('История недоступна')
            state['undo_revision'] = None
        elif action == 'batch':
            operations = operation.get('operations')
            if not isinstance(operations, list) or not 1 <= len(operations) <= 250:
                raise ValueError('Пакет должен содержать от 1 до 250 правок')
            shared_new = None
            for item in operations:
                if not isinstance(item, dict) or item.get('action') in ('undo', 'batch'):
                    raise ValueError('Недопустимое действие внутри пакета')
                item = copy.deepcopy(item)
                if item.get('label') == 'NEW':
                    shared_new = shared_new or _label(state, 'NEW')
                    item['label'] = shared_new
                _apply(d, state, item)
            validate_relations(d, state)
        else:
            _apply(d, state, operation)
            validate_relations(d, state)
        if action != 'undo':
            state['undo_revision'] = old['revision']
        state['revision'] = old['revision'] + 1
        state['last_operation'] = {'action': action, 'at': datetime.now(timezone.utc).isoformat(),
                                   'operation': copy.deepcopy(operation)}
        elapsed = float(operation.get('elapsed_seconds', 0))
        if not math.isfinite(elapsed):
            raise ValueError('Неверное время разметки')
        state['seconds_reviewing'] = old.get('seconds_reviewing', 0) + min(300, max(0, elapsed))
        identity = _identity_view(d, state)  # Validate derived views before the canonical commit.
        atomic_json(d / 'review_history' / ('%08d.json' % old['revision']), old)
        atomic_json(d / 'review_state.json', state)
        materialize(d, state, identity)
        return state


def review_groups(folder, state=None):
    d = Path(folder); state = state or read_state(d)
    originals = read_json(d / ('groups_' + TAG + '.json'), [])
    pieces = {p['piece']: p for p in state['pieces']}
    groups, used = [], set()
    for g in originals:
        ids = set(g['pieces'])
        expanded = [i for i, p in pieces.items() if i in ids or p.get('origin_piece') in ids]
        if expanded:
            groups.append({**g, 'pieces': expanded}); used.update(expanded)
    groups += [{'person': None, 'pieces': [i]} for i in pieces if i not in used]
    for g in groups:
        ps = [pieces[i] for i in g['pieces']]
        labs = [state['labels'].get(str(i)) for i in g['pieces']]
        g.update(detail=ps, t0=min(p['t0'] for p in ps), t1=max(p['t1'] for p in ps),
                 cams=sorted({p['cam'] for p in ps}), dets=sum(len(p['dets']) for p in ps),
                 done=all(lab or state['quality'].get(str(i)) == 'false_positive' for i, lab in zip(g['pieces'], labs)), mixed=len(set(filter(None, labs))) > 1)
    return groups
