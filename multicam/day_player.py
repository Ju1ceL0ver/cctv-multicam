"""Label a whole working day by watching it: the owner's own people, start to finish.

The owner watches both cameras through the day and says who each person is. Nothing is
outlined by hand: the detector already outlined every person on every third frame and the
tracker chained those outlines into continuous tracks (2874 of them on 17.09). One click
selects a whole track, one key says who it is, and the answer covers every frame of it.
Where the tracker put two people into one track, the owner cuts it at the moment it
switches and names both halves.

The store is its own file, `data/day_people/<day>.json`, next to the machine's labels and
not inside them: these are the owner's day-long identities, made without any proposal on
screen, and nothing the night passes do can overwrite them.

Time is the shop clock of camera 1. A detection's clip frame is walked into its raw
segment exactly as the detector read it, and its true moment comes from the recording's
own timestamps (day_proxy sidecars) -- not from counting 25 a second, which falls behind by
up to 14 s inside one window. Camera 2 sees the same moment `cam1_to_cam2_s` later.
"""
import copy
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from threading import RLock

import numpy as np

import day_features
from review_store import read_state
from storage import atomic_json, file_lock, read_json

ROOT = Path(__file__).resolve().parent
FPS = 25
STEP = 0.04
SEGMENT = 900
TAG = 'yolo26x-seg'
SPECIAL = {'_passer': 'прохожий', '_false': 'не человек'}
KINDS = ('customer', 'staff')
_cache = {}
_lock = RLock()


# ---------------------------------------------------------------- time

class Segment:
    def __init__(self, cam, name, start, frames=None, breaks=None, ready=False, origin=0.0, broken=False):
        self.cam, self.name, self.start = cam, name, start          # start: epoch seconds
        self.frames = int(frames) if frames else SEGMENT * FPS
        self.breaks = breaks or [[0, 0.0]]
        self.ready, self.origin, self.broken = ready, float(origin), broken
        self._times = None

    def times(self):
        """Seconds from the file's start for every frame, from the recording's own stamps."""
        if self._times is None:
            out = np.empty(self.frames, np.float64)
            for n, (k, t) in enumerate(self.breaks):
                end = self.breaks[n + 1][0] if n + 1 < len(self.breaks) else self.frames
                out[k:end] = t + STEP * np.arange(end - k)
            self._times = out
        return self._times


def _epoch(stamp):
    return stamp.timestamp() if isinstance(stamp, datetime) else datetime.fromisoformat(stamp).timestamp()


def segments(day, root=ROOT, raw=None):
    """Every raw segment of the day per camera, with its true timing where a copy exists."""
    root = Path(root)
    found = defaultdict(dict)
    for side in sorted((root / 'data/day_proxy' / day).glob('cam*/*.json')):
        record = read_json(side, {})
        if record.get('error'):          # the recorder never finished writing this one
            found[record['cam']][record['raw']] = Segment(
                record['cam'], record['raw'], _epoch(record['start']), broken=True)
        elif record.get('frames'):
            found[record['cam']][record['raw']] = Segment(
                record['cam'], record['raw'], _epoch(record['start']), record['frames'],
                record['breaks'], ready=(side.with_suffix('.mp4')).exists(),
                origin=record['breaks'][0][1] if record.get('breaks') else 0.0)
    for cam in ('cam1', 'cam2'):
        try:
            listed = raw(cam, day) if raw else _raw_segments(cam, day)
        except Exception:                                   # no recordings on this machine
            listed = []
        for path, start in listed:
            name = Path(path).name
            if name not in found[cam]:
                found[cam][name] = Segment(cam, name, _epoch(start))
    return {cam: sorted(found[cam].values(), key=lambda s: s.start) for cam in ('cam1', 'cam2')}


def _raw_segments(cam, day):
    from rawsource import segments as listed
    return listed(cam, day)


def walls(segs, clip_start, frames, landed=None):
    """True epoch seconds of clip frames, walked through the segments the way the detector
    read them: from the frame it started on in the segment covering the clip start, then on
    into the next segments with their *real* frame counts. NaN where no segment covers it.

    `landed` is the file index the detector really started on (see start_frames). Without
    it the nominal index is assumed -- which is wrong wherever the camera had dropped frames
    before the clip start: OpenCV's CAP_PROP_POS_FRAMES seeks by timestamp and lands neither
    on the k-th frame nor on the frame at k/25 s (17.09: 15 of 60 clips on camera 1 off by
    up to 10 s)."""
    frames = np.asarray(frames, np.int64)
    out = np.full(len(frames), np.nan)
    first = next((i for i, s in enumerate(segs) if s.start <= clip_start < s.start + SEGMENT), None)
    if first is None or not len(frames):
        return out
    begin = int(round((clip_start - segs[first].start) * FPS)) if landed is None else int(landed)
    index = begin + frames
    offset = 0
    for seg in segs[first:]:
        local = index - offset
        inside = (local >= 0) & (local < seg.frames)
        if inside.any():
            out[inside] = seg.start + seg.times()[local[inside]]
        offset += seg.frames
        if offset > index.max():
            break
    return out


def start_frames(folder, meta, segs, day, raw=None):
    """{cam: file index the detector really started the clip on, or None if unknown}.

    rawsource.Stream.seek asks OpenCV for frame k = the nominal 25 fps index of the clip start.
    OpenCV turns that into a timestamp seek and then counts frames from the timestamps, so in
    a segment where the camera dropped frames it lands somewhere else -- a test file with 50
    frames missing at 500 lands on 568 when asked for 600. The landing is replayed here once,
    on the raw file with the same OpenCV, and kept next to the clip: raw recordings are pruned
    after two weeks, the answer must outlive them."""
    import cv2
    path = Path(folder) / 'start_frames.json'
    known = read_json(path, {})
    start = _epoch(meta['start'])
    changed = False
    for cam in ('cam1', 'cam2'):
        if cam in known:
            continue
        seg = next((s for s in segs[cam] if s.start <= start < s.start + SEGMENT), None)
        if seg is None or seg.broken or not seg.ready:     # true times come with the light copy
            continue
        try:
            listed = raw(cam, day) if raw else _raw_segments(cam, day)
        except Exception:
            listed = []
        source = next((p for p, _ in listed if Path(p).name == seg.name), None)
        if source is None or not Path(source).exists():
            continue
        nominal = int(round((start - seg.start) * FPS))
        capture = cv2.VideoCapture(str(source))
        try:
            if nominal:
                capture.set(cv2.CAP_PROP_POS_FRAMES, nominal)
            ok, _ = capture.read()
            stamp = capture.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        finally:
            capture.release()
        if not ok:
            continue
        times = seg.times()
        landed = int(np.argmin(np.abs(times - stamp)))
        if abs(times[landed] - stamp) > 0.02:
            continue
        known[cam] = {'segment': seg.name, 'nominal': nominal, 'landed': landed,
                      'shift_s': round(float(times[min(nominal, len(times) - 1)] - times[landed]), 3),
                      'opencv': cv2.__version__}
        changed = True
    if changed:
        atomic_json(path, known)
    return {cam: known[cam]['landed'] if cam in known else None for cam in ('cam1', 'cam2')}


def camera_offset(day, root=ROOT):
    return float(read_json(Path(root) / 'data/cam_sync.json', {}).get(day, {}).get('cam1_to_cam2_s', 0.0))


# ---------------------------------------------------------------- tracks

def _day_clips(day, root):
    for folder in sorted((Path(root) / 'data/raw_clips').glob('c*')):
        meta = read_json(folder / ('meta_%s.json' % TAG), {})
        if meta.get('day') == day:
            yield folder, meta


def _fingerprint(piece):
    dets = piece['dets']
    return '%s:%d:%d:%d' % (piece['cam'], len(dets), dets[0] if dets else -1, dets[-1] if dets else -1)


def tracks(day, root=ROOT):
    """Every machine track of the day, with its true first and last moment on the shop clock
    and every detection box on it. Cached until a clip or a timing sidecar changes."""
    root = Path(root)
    signature = []
    for folder, meta in _day_clips(day, root):
        for name in ('review_state.json', 'pieces_%s.json' % TAG, 'dets_%s.npz' % TAG, 'start_frames.json'):
            path = folder / name
            if path.exists():
                signature.append((folder.name, name, path.stat().st_mtime_ns, path.stat().st_size))
    signature.extend(sorted(p.name + str(p.stat().st_mtime_ns)
                            for p in (root / 'data/day_proxy' / day).glob('cam*/*.json')))
    signature.append(camera_offset(day, root))
    key = (day, str(root))
    with _lock:
        cached = _cache.get(key)
    if cached and cached[0] == signature:
        return cached[1]
    segs = segments(day, root)
    offset = camera_offset(day, root)
    rows, boxes, windows = [], {}, []
    for folder, meta in _day_clips(day, root):
        state = read_state(folder)
        if not state['pieces'] or not (folder / ('dets_%s.npz' % TAG)).exists():
            continue
        start = _epoch(meta['start'])
        windows.append({'clip': folder.name, 'start': start, 'end': start + float(meta['seconds'])})
        with np.load(folder / ('dets_%s.npz' % TAG)) as archive:
            dets = {cam: archive[cam] for cam in ('cam1', 'cam2')}
        landed = start_frames(folder, meta, segs, day)
        clip_boxes = {}
        for piece in state['pieces']:
            cam = piece['cam']
            rows_here = dets[cam][piece['dets']] if piece['dets'] else np.zeros((0, 10))
            if not len(rows_here):
                continue
            frames = np.rint(rows_here[:, 0] * FPS).astype(np.int64)
            when = walls(segs[cam], start, frames, landed[cam]) - (offset if cam == 'cam2' else 0.0)
            good = ~np.isnan(when)
            if not good.any():
                continue
            order = np.argsort(when[good])
            samples = np.column_stack([when[good][order], rows_here[good][order][:, 1:5]])
            track = '%s:%d' % (folder.name, piece['piece'])
            clip_boxes[track] = [[round(float(r[0]), 2)] + [int(round(v)) for v in r[1:5]] for r in samples]
            rows.append({'key': track, 'clip': folder.name, 'piece': piece['piece'], 'cam': cam,
                         'first': round(float(samples[0, 0]), 2), 'last': round(float(samples[-1, 0]), 2),
                         'n': int(len(samples)), 'fp': _fingerprint(piece), 'floor': None})
        boxes[folder.name] = clip_boxes
    # How much of the track stood on the shop floor: the gallery behind the glass is 87% of
    # a day's tracks and none of the shop, and the owner should never be asked about it.
    measured = day_features.load(day, root)
    if measured:
        for row in rows:
            at = measured['index'].get(row['key'])
            if at is not None:
                row['floor'] = round(float(measured['floor'][at]), 3)
    rows.sort(key=lambda r: (r['first'], r['key']))
    result = {'tracks': rows, 'boxes': boxes, 'windows': windows, 'segments': segs, 'offset': offset,
              'measured': bool(measured)}
    with _lock:
        if key not in _cache and len(_cache) >= 4:
            _cache.pop(next(iter(_cache)))
        _cache[key] = (signature, result)
    return result


# ---------------------------------------------------------------- the owner's store

def _store_path(day, root):
    return Path(root) / 'data/day_people' / ('%s.json' % day)


def _empty():
    return {'revision': 0, 'people': {}, 'next': 1, 'tracks': {}, 'undo_revision': None}


def store(day, root=ROOT):
    return read_json(_store_path(day, root), _empty())


def _parts(record):
    return len(record.get('cuts', [])) + 1


def change(day, body, root=ROOT):
    """One edit by the owner. Every edit names the revision it was made against; the old
    state goes to history first, so undo always has something to go back to."""
    root = Path(root)
    path = _store_path(day, root)
    action = body.get('action')
    with file_lock(path.with_suffix('.lock')):
        current = read_json(path, _empty())
        if body.get('revision') != current['revision']:
            raise ValueError('Разметка изменилась в другой вкладке; обновите страницу')
        history = path.parent / 'history' / ('%s_%08d.json' % (day, current['revision']))
        before = copy.deepcopy(current)      # the snapshot undo returns to, taken before any edit
        if action == 'undo':
            back = current.get('undo_revision')
            previous = read_json(path.parent / 'history' / ('%s_%08d.json' % (day, back))) if back is not None else None
            if previous is None:
                raise ValueError('Отменять нечего')
            atomic_json(history, current)
            previous['revision'] = current['revision'] + 1
            previous['undo_revision'] = None
            atomic_json(path, previous)
            return {'revision': previous['revision'], 'store': previous, 'can_undo': False}
        known = {r['key']: r for r in tracks(day, root)['tracks']}
        changed = {}
        if action == 'person':
            name = str(body.get('name') or '').strip()[:60]
            kind = body.get('kind', 'customer')
            if kind not in KINDS or not name:
                raise ValueError('У человека должно быть имя и вид: покупатель или сотрудник')
            ident = body.get('id')
            if ident is None:
                ident = 'P%d' % current['next']
                current['next'] += 1
                current['people'][ident] = {'name': name, 'kind': kind, 'color': (current['next'] - 2) % 12}
            elif ident in current['people']:
                current['people'][ident].update(name=name, kind=kind)
            else:
                raise ValueError('Нет такого человека')
        elif action in ('assign', 'assign_many'):
            who = body.get('who')
            if who is not None and who not in current['people'] and who not in SPECIAL:
                raise ValueError('Нет такого человека')
            pairs = body.get('parts') if action == 'assign_many' else [[body.get('track'), body.get('part', 0)]]
            if not isinstance(pairs, list) or not pairs or len(pairs) > 2000:
                raise ValueError('Не выбраны треки')
            for track, part in pairs:
                if track not in known:
                    raise ValueError('Трек %s больше не существует; обновите страницу' % track)
                record = current['tracks'].setdefault(track, {'cuts': [], 'who': [None], 'fp': known[track]['fp']})
                if record.get('fp') != known[track]['fp']:
                    raise ValueError('Трек %s пересчитан ночью; его разметку надо проверить заново' % track)
                if not isinstance(part, int) or not 0 <= part < _parts(record):
                    raise ValueError('Нет такой части трека')
                record['who'][part] = who
                changed[track] = record
            if body.get('rule'):
                for track, part in pairs:
                    current['tracks'][track].setdefault('rules', {})[str(part)] = str(body['rule'])[:40]
        elif action == 'cut':
            track, at = body.get('track'), body.get('at')
            if track not in known or not isinstance(at, (int, float)):
                raise ValueError('Не выбран трек или момент разреза')
            info = known[track]
            if not info['first'] < at < info['last']:
                raise ValueError('Резать можно только внутри трека')
            record = current['tracks'].setdefault(track, {'cuts': [], 'who': [None], 'fp': info['fp']})
            if record.get('fp') != info['fp']:
                raise ValueError('Трек %s пересчитан ночью; его разметку надо проверить заново' % track)
            if any(abs(at - c) < 0.2 for c in record['cuts']):
                raise ValueError('Здесь уже есть разрез')
            part = sum(1 for c in record['cuts'] if c < at)
            record['cuts'].insert(part, round(float(at), 2))
            record['who'].insert(part + 1, record['who'][part])   # both halves start as they were
            changed[track] = record
        elif action == 'sweep_gallery':
            # Every short track that never set foot on the shop floor, in one go. The rule is
            # written next to each answer, so what was swept can always be told apart later.
            longest = float(body.get('seconds', 10))
            swept = [t for t in known.values() if t.get('floor') == 0 and t['last'] - t['first'] < longest
                     and not current['tracks'].get(t['key'], {}).get('who', [None])[0]]
            if not swept:
                raise ValueError('Таких треков не осталось')
            for track in swept:
                item = current['tracks'].setdefault(track['key'], {'cuts': [], 'who': [None], 'fp': track['fp']})
                if item.get('fp') != track['fp'] or len(item['who']) != 1 or item['who'][0]:
                    continue
                item['who'][0] = '_passer'
                item.setdefault('rules', {})['0'] = 'gallery_not_on_floor'
                changed[track['key']] = item
        else:
            raise ValueError('Неизвестное действие')
        atomic_json(history, before)
        current['undo_revision'] = current['revision']
        current['revision'] += 1
        atomic_json(path, current)
    return {'revision': current['revision'], 'people': current['people'], 'changed': changed,
            'next': current['next'], 'can_undo': True}


# ---------------------------------------------------------------- what the page loads

ACCEPT = 0.228          # the distance auto-linking was calibrated to on 17.09


def similar(day, anchor, root=ROOT, limit=200):
    """Every other unassigned track that could be this same person, nearest look first.

    The owner names somebody once and confirms this grid; that is one screen instead of
    hunting the same person through forty tracks. The order is the machine's opinion, the
    ticks are the machine's guess, and every one of them is a picture the owner looks at.
    A candidate seen at the same moment on the *same* camera is marked impossible: one
    person is not in two places at once.
    """
    data = tracks(day, root)
    measured = day_features.load(day, root)
    owner = store(day, root)
    rows = {t['key']: t for t in data['tracks']}
    if not measured:
        return {'ready': False, 'candidates': []}
    taken = {k: r for k, r in owner['tracks'].items() if any(r['who'])}
    if anchor in owner['people']:
        anchors = [k for k, r in taken.items() if anchor in r['who']]
    else:
        anchors = [anchor] if anchor in rows else []
    anchors = [k for k in anchors if k in measured['index']]
    if not anchors:
        return {'ready': True, 'candidates': [], 'anchors': []}
    vectors = measured['feature'][[measured['index'][k] for k in anchors]]
    spans = [(rows[k]['cam'], rows[k]['first'], rows[k]['last']) for k in anchors]
    out = []
    for key, track in rows.items():
        if key in anchors or key in taken or key not in measured['index']:
            continue
        feature = measured['feature'][measured['index'][key]]
        if not feature.any():
            continue
        distance = float(1 - (vectors @ feature).max())
        clash = any(cam == track['cam'] and first < track['last'] and track['first'] < last
                    for cam, first, last in spans)
        out.append({'key': key, 'clip': track['clip'], 'piece': track['piece'], 'cam': track['cam'],
                    'first': track['first'], 'last': track['last'], 'floor': track['floor'],
                    'distance': round(distance, 4), 'impossible': clash,
                    'suggested': bool(distance <= ACCEPT and not clash)})
    out.sort(key=lambda r: (r['impossible'], r['distance']))
    return {'ready': True, 'anchors': anchors, 'accept': ACCEPT, 'candidates': out[:limit]}


def overview(day, root=ROOT):
    data = tracks(day, root)
    owner = store(day, root)
    prints = {t['key']: t['fp'] for t in data['tracks']}
    stale = [k for k, r in owner['tracks'].items() if k in prints and prints[k] != r.get('fp')]
    return {'day': day, 'offset': data['offset'], 'windows': data['windows'],
            'segments': {cam: [{'name': s.name, 'start': s.start, 'origin': s.origin,
                                'duration': round(s.frames * STEP, 1), 'ready': s.ready, 'broken': s.broken,
                                'url': '/day-proxy/%s/%s/%s' % (day, cam, s.name)}
                               for s in segs] for cam, segs in data['segments'].items()},
            'tracks': data['tracks'], 'store': owner, 'stale': stale,
            'measured': data.get('measured', False),
            'can_undo': owner.get('undo_revision') is not None}


def clip_boxes(day, clip, root=ROOT):
    return tracks(day, root)['boxes'].get(clip, {})


def register(app, root):
    """`root` may be a callable, read on every request like the other routes of the app."""
    import re
    from flask import abort, jsonify, render_template, request, send_file
    here = root if callable(root) else (lambda: root)

    def valid_day(day):
        if not re.fullmatch(r'\d{8}', day):
            abort(400)

    @app.get('/day')
    def day_page():
        return render_template('day.html')

    @app.get('/api/dayplayer/<day>')
    def day_overview(day):
        valid_day(day)
        try:
            return jsonify(overview(day, Path(here())))
        except (ValueError, KeyError, OSError) as exc:
            return jsonify({'error': str(exc)}), 400

    @app.get('/api/dayplayer/<day>/boxes')
    def day_boxes(day):
        valid_day(day)
        clip = request.args.get('clip', '')
        if not re.fullmatch(r'c[A-Za-z0-9_-]+', clip):
            abort(400)
        return jsonify(clip_boxes(day, clip, Path(here())))

    @app.get('/api/dayplayer/<day>/similar')
    def day_similar(day):
        valid_day(day)
        anchor = request.args.get('anchor', '')
        try:
            return jsonify(similar(day, anchor, Path(here())))
        except (ValueError, KeyError, OSError) as exc:
            return jsonify({'error': str(exc)}), 400

    @app.post('/api/dayplayer/<day>')
    def day_change(day):
        valid_day(day)
        try:
            return jsonify(change(day, request.get_json(force=True), Path(here())))
        except (ValueError, KeyError, TypeError) as exc:
            return jsonify({'error': str(exc)}), 409

    @app.get('/day-proxy/<day>/<cam>/<name>')
    def day_proxy_file(day, cam, name):
        valid_day(day)
        if cam not in ('cam1', 'cam2') or not re.fullmatch(r'\d{6}_\d{4}\.mp4', name):
            abort(404)
        path = Path(here()) / 'data/day_proxy' / day / cam / name
        if not path.exists():
            abort(404)
        response = send_file(path, conditional=True, mimetype='video/mp4')
        response.headers['Cache-Control'] = 'private, max-age=86400'
        return response
