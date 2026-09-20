"""Exact-source-frame human corrections over immutable teacher observations.

Intervals include both endpoints. A frame review certifies scene completeness only;
identity, mask quality and accepted propagation remain separate decisions.
"""
import copy
import hashlib
import json
import math
import re
import uuid
from collections import OrderedDict
from pathlib import Path
from threading import RLock

import numpy as np
from storage import read_json

FPS = 25
TAG = 'yolo26x-seg'
CAMS = ('cam1', 'cam2')
VIDEO_ACTIONS = {'label_interval', 'quality_interval', 'swap_interval',
                 'upsert_observation', 'delete_observation', 'review_frame'}
QUALITY = {'unreviewed', 'valid', 'invalid', 'false_positive', 'mixed'}
_CACHE = OrderedDict()
_CACHE_LOCK = RLock()
_PIECE_CACHE = OrderedDict()


def video_state(state):
    return state.setdefault('video', {'version': 1, 'observations': {},
                                      'intervals': [], 'frame_reviews': {}})


def _integer(value, name):
    if isinstance(value, bool):
        raise ValueError('Неверное целое число: ' + name)
    try:
        result = int(value)
        if float(value) != result:
            raise ValueError()
    except (ValueError, TypeError, OverflowError):
        raise ValueError('Неверное целое число: ' + name)
    return result


def _assets(folder):
    folder = Path(folder)
    names = [f'{prefix}_{TAG}.npz' for prefix in ('dets', 'polys', 'grid')]
    signature = tuple((name, (folder/name).stat().st_mtime_ns, (folder/name).stat().st_size)
                      for name in names if (folder/name).exists())
    key = str(folder.resolve())
    with _CACHE_LOCK:
        cached = _CACHE.get(key)
        if cached and cached[0] == signature:
            _CACHE.move_to_end(key)
            return cached[1]
        path = folder / names[0]
        arrays = {}
        if path.exists():
            with np.load(path) as archive:
                arrays = {cam: archive[cam].copy() for cam in CAMS}
        for cam in CAMS:
            arrays.setdefault(cam, np.empty((0, 10)))
        polygons = {}
        if (folder/names[1]).exists():
            with np.load(folder/names[1]) as archive:
                polygons = {k: archive[k].copy() for k in archive.files}
        by_frame = {}
        for cam, rows in arrays.items():
            index = {}
            for i, frame in enumerate(np.rint(rows[:, 0] * FPS).astype(np.int64)):
                index.setdefault(int(frame), []).append(i)
            by_frame[cam] = index
        sampled = None
        if (folder/names[2]).exists():
            with np.load(folder/names[2]) as archive:
                sampled = set(np.rint(archive['t'] * FPS).astype(int).tolist())
        result = {'rows': arrays, 'polygons': polygons, 'by_frame': by_frame,
                  'sampled': sampled, 'signature': signature}
        _CACHE[key] = (signature, result)
        _CACHE.move_to_end(key)
        while len(_CACHE) > 2:
            _CACHE.popitem(last=False)
        return result


def dimensions(folder, cam):
    folder = Path(folder)
    cached = read_json(folder/'review_media/source.json', {})
    value = cached.get('dimensions', {}).get(cam)
    meta = read_json(folder/f'meta_{TAG}.json', {})
    if value is None:
        value = meta.get('dimensions', {}).get(cam)
    if value is None and meta.get('width') and meta.get('height'):
        value = [meta['width'], meta['height']]
    if value is None:
        value = [2560, 1440]
    w, h = (_integer(v, 'dimensions') for v in value)
    if w < 1 or h < 1:
        raise ValueError('Неверный размер исходного кадра')
    return w, h


def _frame(folder, cam, frame):
    if cam not in CAMS:
        raise ValueError('Неизвестная камера')
    frame = _integer(frame, 'frame')
    meta = read_json(Path(folder)/f'meta_{TAG}.json', {})
    end = int(math.ceil(float(meta['seconds']) * FPS)) if meta.get('seconds') is not None else None
    if frame < 0 or (end is not None and frame >= end):
        raise ValueError('Кадр вне записи')
    return frame


def _piece_map(state):
    key = id(state)
    with _CACHE_LOCK:
        cached = _PIECE_CACHE.get(key)
        if cached and cached[0] is state and cached[1] == len(state['pieces']):
            return cached[2]
    result = {}
    for piece in state['pieces']:
        for i in piece['dets']:
            key = (piece['cam'], int(i))
            if key in result and result[key] != int(piece['piece']):
                raise ValueError('Одна детекция принадлежит двум отрезкам')
            result[key] = int(piece['piece'])
    with _CACHE_LOCK:
        _PIECE_CACHE[id(state)] = (state, len(state['pieces']), result)
        _PIECE_CACHE.move_to_end(id(state))
        while len(_PIECE_CACHE) > 3:
            _PIECE_CACHE.popitem(last=False)
    return result


def _resolve(state, piece, frame, kind, default=None):
    value = default
    if piece is not None:
        value = state['labels' if kind == 'label' else 'quality'].get(str(piece), default)
        for interval in state.get('video', {}).get('intervals', []):
            if (interval['kind'] == kind and interval['piece'] == piece and
                    interval['start_frame'] <= frame <= interval['end_frame']):
                value = interval['value']
    return value


def effective_decision(state, piece, raw_frame, observation_id=None):
    """Cheap identity/quality resolver for derived views, with optional exact override.

    Pass det:cam:index for an existing observation or manual:uuid for a new one.
    A deleted observation is returned as false_positive; it is never identity GT.
    """
    video = state.get('video', {})
    override = video.get('observations', {}).get(observation_id, {})
    if override.get('deleted'):
        return None, 'false_positive'
    if piece is None:
        piece = override.get('piece')
    result = {'label': _resolve(state, piece, raw_frame, 'label'),
              'quality': _resolve(state, piece, raw_frame, 'quality', 'unreviewed')}
    for kind in ('label', 'quality'):
        if kind in override:
            result[kind] = override[kind]
        for interval in video.get('intervals', []):
            if (piece == interval['piece'] and interval['kind'] == kind and
                    interval['start_frame'] <= raw_frame <= interval['end_frame'] and
                    interval.get('sequence', 0) > override.get(kind+'_sequence', -1)):
                result[kind] = interval['value']
    return result['label'], result['quality']


def _original(folder, cam, index, state, assets=None, pieces=None):
    assets = assets or _assets(folder)
    rows = assets['rows'][cam]
    if index < 0 or index >= len(rows):
        raise ValueError('Неизвестное наблюдение')
    row = rows[index]
    frame = int(round(float(row[0]) * FPS))
    pieces = pieces if pieces is not None else _piece_map(state)
    piece = pieces.get((cam, index))
    polygons = assets['polygons']
    points, offsets = polygons.get(cam+'_pts'), polygons.get(cam+'_off')
    polygon = points[offsets[index]:offsets[index+1]].tolist() if points is not None and offsets is not None else []
    return {'id': f'det:{cam}:{index}', 'cam': cam, 'frame': frame, 'piece': piece,
            'box': row[1:5].astype(float).tolist(), 'polygon': polygon,
            'label': _resolve(state, piece, frame, 'label'),
            'quality': _resolve(state, piece, frame, 'quality', 'unreviewed'),
            'source': 'teacher', 'mask_review_source': 'teacher_proposal',
            'detection_index': index}


def _observation(folder, state, oid, assets=None, pieces=None):
    override = state.get('video', {}).get('observations', {}).get(oid)
    match = re.fullmatch(r'det:(cam[12]):(\d+)', str(oid))
    if match:
        result = _original(folder, match[1], int(match[2]), state, assets, pieces)
    elif override is not None:
        result = {'id': oid, 'piece': None, 'polygon': [], 'quality': 'unreviewed',
                  'label': None, 'mask_review_source': 'unreviewed'}
    else:
        raise ValueError('Неизвестное наблюдение')
    if override:
        result.update(copy.deepcopy(override))
        for kind, default in (('label', None), ('quality', 'unreviewed')):
            if kind not in override:
                result[kind] = _resolve(state, result.get('piece'), result['frame'], kind, default)
    # Interval edits have precedence over an older per-frame label/quality.
    for interval in state.get('video', {}).get('intervals', []):
        if (result.get('piece') == interval['piece'] and
                interval['start_frame'] <= result['frame'] <= interval['end_frame'] and
                interval.get('sequence', 0) > result.get(interval['kind']+'_sequence', -1)):
            result[interval['kind']] = interval['value']
            if interval['kind'] == 'quality':
                result['mask_review_source'] = ('human_accepted_interval' if interval['value'] == 'valid' else 'unreviewed')
    return result


def _completeness_evidence(observations):
    evidence = [{k: obs.get(k) for k in ('id', 'box', 'polygon', 'mask_rle', 'source')}
                for obs in observations if obs.get('quality') != 'false_positive']
    return hashlib.sha256(json.dumps(evidence, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def frame_data(folder, cam, frame, state=None):
    """Return exact-frame observations, never nearest-frame teacher ghosts."""
    if state is None:
        from review_store import read_state
        state = read_state(folder)
    frame = _frame(folder, cam, frame)
    width, height = dimensions(folder, cam)
    assets = _assets(folder)
    pieces = _piece_map(state)
    ids = [f'det:{cam}:{i}' for i in assets['by_frame'][cam].get(frame, [])]
    for oid, item in state.get('video', {}).get('observations', {}).items():
        if item.get('cam') == cam and item.get('frame') == frame and oid not in ids:
            ids.append(oid)
    observations = [_observation(folder, state, oid, assets, pieces) for oid in ids]
    observations = [obs for obs in observations if not obs.get('deleted')]
    evidence = _completeness_evidence(observations)
    approved = state.get('video', {}).get('frame_reviews', {}).get(f'{cam}:{frame}', {})
    sampled = frame in assets['sampled'] if assets['sampled'] is not None else frame in assets['by_frame'][cam]
    return {'cam': cam, 'frame': frame, 'fps': FPS, 'width': width, 'height': height,
            'revision': state['revision'], 'observations': observations,
            'teacher_sampled': sampled,
            'frame_reviewed': bool(approved.get('reviewed') and approved.get('evidence') == evidence),
            'frame_evidence': evidence}


def effective_frame(folder, cam, frame, state=None):
    return frame_data(folder, cam, frame, state=state)


def _label(state, value):
    if value == 'NEW':
        used = list(state['labels'].values())
        used.extend(o.get('label') for o in state.get('video', {}).get('observations', {}).values())
        used.extend(i['value'] for i in state.get('video', {}).get('intervals', []) if i['kind'] == 'label')
        numbers = [int(v[1:]) for v in used if isinstance(v, str) and re.fullmatch(r'P\d+', v)]
        return 'P%d' % (max(numbers, default=0) + 1)
    if value is not None and (not isinstance(value, str) or not value or len(value) > 60):
        raise ValueError('Неверная метка')
    return value


def _interval(folder, state, operation):
    piece = _integer(operation['piece'], 'piece')
    p = next((p for p in state['pieces'] if p['piece'] == piece), None)
    if p is None:
        raise ValueError('Неизвестный отрезок')
    start = _frame(folder, p['cam'], operation['start_frame'])
    end = _frame(folder, p['cam'], operation['end_frame'])
    if start > end:
        raise ValueError('Начало интервала позже конца')
    assets = _assets(folder)
    frames = np.rint(assets['rows'][p['cam']][p['dets'], 0] * FPS).astype(int)
    manual_present = any(o.get('piece') == piece and not o.get('deleted') and start <= o['frame'] <= end
                         for oid, o in state.get('video', {}).get('observations', {}).items() if oid.startswith('manual:'))
    if not np.any((frames >= start) & (frames <= end)) and not manual_present:
        raise ValueError('В выбранном интервале нет наблюдений этого человека')
    return piece, p, start, end


def _add_interval(state, kind, piece, start, end, value):
    video = video_state(state)
    seq = video.get('sequence', 0) + 1
    video['sequence'] = seq
    video['intervals'].append({'kind': kind, 'piece': piece, 'start_frame': start,
                              'end_frame': end, 'value': value, 'sequence': seq})


def clear_intervals(state, kind, pieces):
    video = state.get('video')
    if not video:
        return
    video['intervals'] = [i for i in video['intervals'] if not (i['kind'] == kind and i['piece'] in pieces)]
    for obs in video['observations'].values():
        if obs.get('piece') in pieces:
            obs.pop(kind, None)
            obs.pop(kind+'_sequence', None)


def _finite_points(value, width, height, box=False):
    try:
        array = np.asarray(value, dtype=float)
    except (ValueError, TypeError):
        raise ValueError('Неверные координаты')
    if box:
        if array.shape != (4,):
            raise ValueError('Рамка должна содержать четыре координаты')
        if not (array[0] < array[2] and array[1] < array[3]):
            raise ValueError('Рамка должна иметь положительную площадь')
        points = array.reshape(2, 2)
    else:
        if array.ndim != 2 or array.shape[1] != 2 or not 3 <= len(array) <= 20000:
            raise ValueError('Контур должен содержать не менее трёх точек')
        points = array
        area = abs(float(np.dot(points[:, 0], np.roll(points[:, 1], 1)) - np.dot(points[:, 1], np.roll(points[:, 0], 1)))) / 2
        if area <= 0:
            raise ValueError('Контур должен иметь положительную площадь')
    if not np.isfinite(points).all() or np.any(points < 0) or np.any(points[:, 0] > width) or np.any(points[:, 1] > height):
        raise ValueError('Координаты вне исходного кадра')
    return array.tolist()


def validate_rle(value, width, height):
    if not isinstance(value, dict) or value.get('size') != [height, width]:
        raise ValueError('RLE должен иметь размер исходного кадра [height,width]')
    counts = value.get('counts')
    if not isinstance(counts, list) or not counts or len(counts) > width * height + 1:
        raise ValueError('Поддерживается несжатый COCO RLE: counts должен быть списком')
    if any(isinstance(c, bool) or not isinstance(c, int) or c < 0 for c in counts) or sum(counts) != width * height:
        raise ValueError('Неверные длины RLE')
    if sum(counts[1::2]) == 0:
        raise ValueError('Маска пуста; используйте удаление наблюдения')
    return {'size': [height, width], 'counts': counts.copy()}


def apply_operation(folder, state, operation):
    """Mutate a transaction working copy; caller owns lock/version/history."""
    video = video_state(state)
    action = operation['action']
    if action in ('label_interval', 'quality_interval'):
        piece, p, start, end = _interval(folder, state, operation)
        kind = action.split('_')[0]
        value = _label(state, operation.get('label')) if kind == 'label' else operation.get('quality')
        if kind == 'quality' and value not in QUALITY:
            raise ValueError('Неизвестное состояние маски')
        _add_interval(state, kind, piece, start, end, value)
    elif action == 'swap_interval':
        a, pa, start, end = _interval(folder, state, operation)
        b = _integer(operation['other_piece'], 'other_piece')
        pb = next((p for p in state['pieces'] if p['piece'] == b), None)
        if pb is None or a == b or pa['cam'] != pb['cam']:
            raise ValueError('Для обмена выберите два отрезка одной камеры')
        labels = []
        assets = _assets(folder)
        for p in (pa, pb):
            values = set()
            for i in p['dets']:
                frame = int(round(float(assets['rows'][p['cam']][i, 0]) * FPS))
                if start <= frame <= end:
                    obs = _observation(folder, state, f'det:{p["cam"]}:{i}', assets)
                    if not obs.get('deleted'):
                        values.add(obs.get('label'))
            for oid, item in video['observations'].items():
                if oid.startswith('manual:') and item.get('piece') == p['piece'] and start <= item['frame'] <= end and not item.get('deleted'):
                    values.add(_observation(folder, state, oid, assets).get('label'))
            if len(values) != 1 or next(iter(values), None) in (None, '?'):
                raise ValueError('В каждом выбранном интервале должна быть одна известная личность')
            labels.append(values.pop())
        if labels[0] == labels[1]:
            raise ValueError('У выбранных интервалов уже один номер')
        _add_interval(state, 'label', a, start, end, labels[1])
        _add_interval(state, 'label', b, start, end, labels[0])
    elif action == 'upsert_observation':
        cam = operation.get('cam')
        frame = _frame(folder, cam, operation.get('frame'))
        width, height = dimensions(folder, cam)
        oid = operation.get('id')
        existing = None
        if oid is not None:
            if not isinstance(oid, str):
                raise ValueError('Неверный идентификатор наблюдения')
            existing = _observation(folder, state, oid)
            if existing['cam'] != cam or existing['frame'] != frame:
                raise ValueError('Нельзя перенести существующее наблюдение на другой исходный кадр')
        else:
            oid = 'manual:' + uuid.uuid4().hex
        box = _finite_points(operation.get('box'), width, height, box=True)
        item = copy.deepcopy(existing) if existing else {'id': oid, 'piece': None, 'label': None, 'quality': 'unreviewed', 'polygon': []}
        if 'piece' in operation:
            piece = operation['piece']
            if piece is not None:
                piece = _integer(piece, 'piece')
                p = next((p for p in state['pieces'] if p['piece'] == piece and p['cam'] == cam), None)
                if p is None:
                    raise ValueError('Неизвестный отрезок этой камеры')
            if oid.startswith('det:') and piece != item['piece']:
                raise ValueError('Принадлежность исходной детекции меняется через интервальные правки')
            item['piece'] = piece
        source = operation.get('source', 'human_anchor')
        if source not in ('human_anchor', 'sam_propagation'):
            raise ValueError('Неизвестное происхождение правки')
        geometry_changed = box != item.get('box')
        if 'polygon' in operation:
            item['polygon'] = _finite_points(operation['polygon'], width, height)
            item.pop('mask_rle', None)
            geometry_changed = True
        if 'mask_rle' in operation:
            item['mask_rle'] = validate_rle(operation['mask_rle'], width, height)
            if 'polygon' not in operation:
                item['polygon'] = []
            geometry_changed = True
        if geometry_changed or not existing or existing.get('deleted'):
            item['quality'] = 'unreviewed'
        if source == 'human_anchor':
            item.pop('provenance', None)
        item.update(id=oid, cam=cam, frame=frame, box=box, source=source, deleted=False)
        seq = video.get('sequence', 0) + 1
        video['sequence'] = seq
        if 'label' in operation:
            item['label'] = _label(state, operation['label'])
            item['label_sequence'] = seq
        if 'quality' in operation:
            if operation['quality'] not in QUALITY:
                raise ValueError('Неизвестное состояние маски')
            item['quality'] = operation['quality']
        if item['quality'] == 'valid' and source == 'sam_propagation' and operation.get('approval') != 'accepted_propagation':
            raise ValueError('Перенос SAM требует явного подтверждения интервала')
        if item['quality'] == 'valid' and not item.get('polygon') and not item.get('mask_rle'):
            raise ValueError('Нельзя подтвердить маску, которой нет')
        item['quality_sequence'] = seq
        item['mask_review_source'] = ('human_accepted_propagation' if source == 'sam_propagation' and item['quality'] == 'valid'
                                      else 'human_anchor' if source == 'human_anchor' and item['quality'] == 'valid' else 'unreviewed')
        if 'provenance' in operation:
            provenance = operation['provenance']
            if not isinstance(provenance, dict) or any(k not in ('job_id', 'model', 'anchor_id', 'input_revision') for k in provenance):
                raise ValueError('Неверное происхождение предложения')
            if any(not isinstance(v, (str, int)) or isinstance(v, bool) or len(str(v)) > 200 for v in provenance.values()):
                raise ValueError('Неверное происхождение предложения')
            item['provenance'] = copy.deepcopy(provenance)
        video['observations'][oid] = item
        state['_last_observation_id'] = oid
    elif action == 'delete_observation':
        oid = operation['id']
        item = _observation(folder, state, oid)
        item['deleted'] = True
        item['quality'] = 'false_positive'
        item['source'] = 'human_anchor'
        video['observations'][oid] = item
    elif action == 'review_frame':
        cam = operation.get('cam')
        frame = _frame(folder, cam, operation.get('frame'))
        reviewed = operation.get('reviewed')
        if not isinstance(reviewed, bool):
            raise ValueError('Укажите явное подтверждение полноты кадра')
        data = frame_data(folder, cam, frame, state=state)
        video['frame_reviews'][f'{cam}:{frame}'] = {'reviewed': reviewed, 'evidence': data['frame_evidence']}
    else:
        raise ValueError('Неизвестная видеоправка')


def identity_materialization(folder, state):
    """Legacy identity view includes interval decisions, never renumbers detections."""
    assets = _assets(folder)
    pieces = _piece_map(state)
    gt = {cam: {} for cam in CAMS}
    for (cam, i), piece in pieces.items():
        obs = _observation(folder, state, f'det:{cam}:{i}', assets, pieces)
        if obs.get('label') not in (None, '?') and obs.get('quality') not in ('false_positive', 'mixed') and not obs.get('deleted'):
            gt[cam][str(i)] = obs['label']
    return gt


def validate_relations(folder, state):
    """Reject contradictions, including indirect same links and interval labels."""
    pieces = {int(p['piece']) for p in state['pieces']}
    relations = state.get('relations', [])
    if not relations:
        return
    parent = {i: i for i in pieces}
    def find(i):
        while parent[i] != i:
            i = parent[i]
        return i
    for r in relations:
        if r['a'] not in pieces or r['b'] not in pieces:
            raise ValueError('Связь ссылается на отсутствующий отрезок')
        if r['decision'] == 'same':
            parent[find(r['b'])] = find(r['a'])
    labels = {i: set() for i in pieces}
    assets = _assets(folder)
    mapping = _piece_map(state)
    for (cam, i), piece in mapping.items():
        obs = _observation(folder, state, f'det:{cam}:{i}', assets, mapping)
        if obs.get('label') not in (None, '?') and not obs.get('deleted') and obs.get('quality') not in ('false_positive', 'mixed'):
            labels[piece].add(obs['label'])
    for oid, raw in state.get('video', {}).get('observations', {}).items():
        if oid.startswith('manual:') and raw.get('piece') in pieces:
            obs = _observation(folder, state, oid, assets, mapping)
            if obs.get('label') not in (None, '?') and not obs.get('deleted') and obs.get('quality') not in ('false_positive', 'mixed'):
                labels[obs['piece']].add(obs['label'])
    components = {}
    for i, values in labels.items():
        components.setdefault(find(i), set()).update(values)
    for r in relations:
        a, b = find(r['a']), find(r['b'])
        if r['decision'] == 'same' and len(components[a]) > 1:
            raise ValueError('Правка противоречит ответу «тот же человек». Сначала уточните связь или границы.')
        if r['decision'] == 'different' and (a == b or components[a] & components[b]):
            raise ValueError('Правка противоречит ответу «разные люди». Сначала исправьте связь.')


def video_transact(folder, operation, expected=None):
    from review_store import transact
    return transact(folder, operation, expected)
