"""The working day as a film: one long video per camera, on one clock, with everybody in it.

The owner watches the whole day at speed and corrects the machine where he sees it go
wrong. He is never asked who somebody is: he could not tell from a piece of a track, and
he does not need to. The machine makes the people itself -- its grouping inside each
ten-minute window, the same track carried across the seam between two windows, the same
group on both cameras -- and gives each one a number and a colour. The owner only says
"that is the same one" (two clicks), "from here this is somebody else" (S), "not a person"
(X), "staff" (W) or "somebody here has no outline" (M). Whatever he watched without
correcting is recorded as watched.

The films. Each camera gets one H.264 file for the whole day, 640x360 at 12.5 frames a
second, cut from the day's light copies (day_proxy). Frame k of *both* files is the same
moment on the shop clock: every tick takes the last frame the camera had shot by then
(frames the camera never delivered repeat the previous one), and camera 2 is read
`cam1_to_cam2_s` later. So the two players need no synchronising beyond "same position",
and a detection's shop time is its position in the film. A key frame every two seconds
makes a jump land at once. Where a camera recorded nothing the tick is dark, and the
sidecar lists those holes.

Only one clip is shown for any moment. Five early clips were cut at odd times over the
regular ten-minute grid; where the grid covers a moment the early clip's detections are
not drawn, otherwise every person there would carry two outlines.
"""
import copy
import gzip
import json
import math
import os
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from threading import RLock

import numpy as np

import day_features
import day_player
from review_store import read_state
from storage import atomic_json, file_lock, read_json

ROOT = Path(__file__).resolve().parent
TAG = 'yolo26x-seg'
EMB = 'emb_osnet_ain_x1_0_msmt17.npz'
TICK = 0.08                  # 12.5 frames a second: the teacher looked 8.3 times a second
WIDTH, HEIGHT = 640, 360
UNIT = 1280                  # outlines on the wire are in half-resolution pixels of the 2560 frame
KINDS = ('customer', 'staff', 'passer')
GOP = 25                     # a key frame every 2 s of film
# No B-frames: with them the index at the head of a day's file is 5 MB per camera (17.09),
# which the browser must fetch before the first frame -- 5-7 s through the tunnel.
CRF = 30                     # 181 kbit/s on 17.09, 0.73 GB per camera-day (measured)
OUTLINE_VERSION = 3          # bump when clean_outline changes: the cached outlines are rebuilt
EDGE = 0.04                  # a box this close to the frame border came in or went out of view
SHOP_SECONDS = 10.0          # a track this long, or one that set foot on the shop floor, is a visitor
SEAM_IOU = 0.4               # 17.09: 58 of 105 seam tracks link at this overlap, none ambiguous
SEAM_GAP = 1.0
CHANGE = 0.15                # appearance jump that marks where a track changed person
VERSION = 1
_cache = {}
_clip_cache = {}
_lock = RLock()


# ---------------------------------------------------------------- the clock of the film

def offset_of(day, root=ROOT):
    """Seconds camera 2 sees a moment later than camera 1, for the film. The film keeps its own
    value (data/day_movie/<day>/offset.json) when one was measured for it, so correcting it does
    not touch the night pipeline's cam_sync.json; otherwise the pipeline's day value."""
    own = read_json(folder(day, root) / 'offset.json', None)
    if own and isinstance(own.get('cam1_to_cam2_s'), (int, float)):
        return float(own['cam1_to_cam2_s'])
    return day_player.camera_offset(day, root)


def span(day, root=ROOT):
    """(start, end) of the film on the shop clock, the same for both cameras. Nominal
    segment lengths are used on purpose: the span must not move when a light copy appears.
    Camera 2's own span is not used: the film starts where camera 1's recording starts."""
    segs = day_player.segments(day, root)
    offset = offset_of(day, root)
    starts, ends = [], []
    for cam, shift in (('cam1', 0.0), ('cam2', offset)):
        if segs[cam]:
            starts.append(segs[cam][0].start - shift)
            ends.append(segs[cam][-1].start + day_player.SEGMENT - shift)
    if not starts:
        raise ValueError('За этот день нет записей')
    return float(math.floor(min(starts))), float(math.ceil(max(ends)))


def clock(day, root=ROOT):
    """(start, end) of the film as it was built: the first film written fixes the day's clock,
    and every later build and every outline follows it. Rebuilding camera 2 with a corrected
    offset must not move camera 1's film."""
    for cam in ('cam1', 'cam2'):
        record = read_json(film(day, cam, root)[1], None)
        if record and record.get('version') == VERSION and 'start' in record:
            return float(record['start']), float(record['end'])
    return span(day, root)


def folder(day, root=ROOT):
    return Path(root) / 'data/day_movie' / day


def film(day, cam, root=ROOT):
    return folder(day, root) / ('%s.mp4' % cam), folder(day, root) / ('%s.json' % cam)


class Reader:
    """One camera's frames in true time, read once, in order, from the light copies."""

    def __init__(self, day, cam, root=ROOT):
        import cv2
        self.cv2 = cv2
        every = day_player.segments(day, root)[cam]
        self.segs = [s for s in every if s.ready and not s.broken]
        self.missing = [s.name for s in every if not s.ready and not s.broken]
        self.broken = [s.name for s in every if s.broken]
        self.base = Path(root) / 'data/day_proxy' / day / cam
        self.firsts = [s.start + s.times()[0] for s in self.segs]
        self.i, self.capture, self.times, self.k, self.image, self.image_k = -1, None, None, -1, None, -1

    def _open(self, i):
        if self.capture is not None:
            self.capture.release()
        cv2 = self.cv2
        path = str(self.base / self.segs[i].name)
        try:
            self.capture = cv2.VideoCapture(path, cv2.CAP_FFMPEG, [cv2.CAP_PROP_N_THREADS, 2])
        except (TypeError, AttributeError, cv2.error):          # older OpenCV: no open parameters
            self.capture = cv2.VideoCapture(path)
        self.i, self.times = i, self.segs[i].start + self.segs[i].times()
        self.k, self.image, self.image_k = -1, None, -1

    def at(self, when):
        """The last frame shot at or before `when`, 640x360 BGR, or None where nothing was."""
        while self.i + 1 < len(self.segs) and when >= self.firsts[self.i + 1] - 1e-6:
            self._open(self.i + 1)
        if self.i < 0 or when > self.times[-1] + 0.5:
            return None
        target = int(np.searchsorted(self.times, when + 1e-6, 'right')) - 1
        if target < 0:
            return None
        while self.k < target:
            if not self.capture.grab():
                break
            self.k += 1
        if self.image_k != self.k:
            ok, frame = self.capture.retrieve()
            if not ok:
                return self.image
            self.image = self.cv2.resize(frame, (WIDTH, HEIGHT), interpolation=self.cv2.INTER_AREA)
            self.image_k = self.k
        return self.image

    def close(self):
        if self.capture is not None:
            self.capture.release()


def _below_normal():
    if os.name == 'nt':
        import ctypes
        ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), 0x4000)
    else:
        try:
            os.nice(10)
        except OSError:
            pass


def build(day, cam, root=ROOT, ffmpeg=None, threads=2, log=print):
    """One camera's film for the whole day. Written aside and moved in place at the end, so
    the page never sees half a file."""
    from day_proxy import _ffmpeg
    root = Path(root)
    start, end = clock(day, root)
    shift = offset_of(day, root) if cam == 'cam2' else 0.0
    ticks = int(round((end - start) / TICK))
    target, side = film(day, cam, root)
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(cam + '.part.mp4')
    reader = Reader(day, cam, root)
    command = [ffmpeg or _ffmpeg(), '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'bgr24',
               '-s', '%dx%d' % (WIDTH, HEIGHT), '-r', '12.5', '-i', '-', '-an', '-c:v', 'libx264',
               '-preset', 'veryfast', '-crf', str(CRF), '-g', str(GOP), '-keyint_min', str(GOP),
               '-sc_threshold', '0', '-bf', '0', '-pix_fmt', 'yuv420p', '-threads', str(threads),
               '-movflags', '+faststart', str(partial)]
    flags = getattr(subprocess, 'BELOW_NORMAL_PRIORITY_CLASS', 0)
    pipe = subprocess.Popen(command, stdin=subprocess.PIPE, creationflags=flags)
    blank = np.full((HEIGHT, WIDTH, 3), 24, np.uint8)
    holes, opened, began = [], None, time.time()
    try:
        for k in range(ticks):
            image = reader.at(start + k * TICK + shift)
            if image is None:
                if opened is None:
                    opened = k
                image = blank
            elif opened is not None:
                holes.append([round(opened * TICK, 2), round(k * TICK, 2)])
                opened = None
            pipe.stdin.write(image.tobytes())
            if k and k % 9000 == 0:
                log('%s %s: %d / %d frames, %.0f s' % (day, cam, k, ticks, time.time() - began))
        if opened is not None:
            holes.append([round(opened * TICK, 2), round(ticks * TICK, 2)])
    finally:
        reader.close()
        pipe.stdin.close()
        code = pipe.wait()
    if code:
        partial.unlink(missing_ok=True)
        raise RuntimeError('ffmpeg ended with code %d' % code)
    os.replace(partial, target)
    record = {'day': day, 'cam': cam, 'start': start, 'end': end, 'tick': TICK, 'frames': ticks,
              'shift': shift, 'width': WIDTH, 'height': HEIGHT, 'holes': holes, 'bytes': target.stat().st_size,
              'missing': reader.missing, 'broken': reader.broken, 'build_s': round(time.time() - began, 1),
              'version': VERSION}
    atomic_json(side, record)
    log(json.dumps({k: record[k] for k in ('cam', 'frames', 'bytes', 'build_s', 'missing', 'broken')}))
    return record


# ---------------------------------------------------------------- the machine's people

def _clips(day, root):
    for path in sorted((Path(root) / 'data/raw_clips').glob('c*')):
        meta = read_json(path / ('meta_%s.json' % TAG), {})
        if meta.get('day') == day:
            yield path, meta


def _grid(day, root):
    """Start of the regular ten-minute grid the night passes cut (auto_label.windows)."""
    segs = day_player.segments(day, root)
    if not segs['cam1'] or not segs['cam2']:
        return None
    return max(segs['cam1'][0].start, segs['cam2'][0].start) + 30


def _on_grid(start, grid):
    if grid is None:
        return True
    steps = (start - grid) / 600.0
    return abs(steps - round(steps)) * 600 < 2


def _signature(day, root):
    parts = []
    for path, _ in _clips(day, root):
        for name in ('review_state.json', 'pieces_%s.json' % TAG, 'dets_%s.npz' % TAG, 'groups_%s.json' % TAG,
                     'start_frames.json'):
            p = path / name
            if p.exists():
                parts.append((path.name, name, p.stat().st_mtime_ns))
    parts.extend(sorted(p.name + str(p.stat().st_mtime_ns)
                        for p in (Path(root) / 'data/day_proxy' / day).glob('cam*/*.json')))
    feats = day_features.store_path(day, root)
    parts.append(feats.stat().st_mtime_ns if feats.exists() else 0)
    parts.append(offset_of(day, root))
    parts.extend(read_json(film(day, cam, root)[1], {}).get('start') for cam in ('cam1', 'cam2'))
    return parts


def _iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


_floors = {}


def _floor_share(cam, rows):
    """Share of a track's detections whose feet stand on the hand-traced shop floor."""
    if cam not in _floors:
        import rooms
        _floors[cam] = rooms.mask(cam) > 0
    m = _floors[cam]
    x = np.clip(rows[:, 6].astype(int), 0, m.shape[1] - 1)
    y = np.clip(rows[:, 7].astype(int), 0, m.shape[0] - 1)
    return float(m[y, x].mean()) if len(rows) else 0.0


def _leftovers(d, used, gap=0.6, overlap=0.3, duplicate=0.5):
    """Detections outside every piece, chained frame to frame by box overlap. A box lying on a
    box the tracker kept (a second outline on the same person) is not a person of its own."""
    from collections import defaultdict
    kept_at = defaultdict(list)
    for i in used:
        kept_at[round(float(d[i, 0]), 3)].append(i)
    groups = defaultdict(list)
    for i in range(len(d)):
        if i in used:
            continue
        t = round(float(d[i, 0]), 3)
        if any(_iou(d[i, 1:5], d[j, 1:5]) >= duplicate for j in kept_at.get(t, ())):
            continue
        groups[t].append(i)
    chains, live = [], []
    for t in sorted(groups):
        live = [c for c in live if t - d[c[-1], 0] <= gap]
        pairs = sorted(((_iou(d[c[-1], 1:5], d[j, 1:5]), n, j) for n, c in enumerate(live) for j in groups[t]),
                       reverse=True)
        took, placed = set(), set()
        for score, n, j in pairs:
            if score < overlap:
                break
            if n in took or j in placed:
                continue
            live[n].append(j)
            took.add(n)
            placed.add(j)
        for j in groups[t]:
            if j not in placed:
                chains.append([j])
                live.append(chains[-1])
    return chains


def _at_edge(box):
    x1, y1, x2, y2 = box
    return x1 < EDGE * 2560 or x2 > (1 - EDGE) * 2560 or y1 < EDGE * 1440 or y2 > (1 - EDGE) * 1440


def index(day, root=ROOT):
    """Every track the film shows, on the film's clock, with the machine's own links between
    them. Cached until a clip, a timing sidecar or the camera offset changes."""
    root = Path(root)
    signature = _signature(day, root)
    key = (day, str(root))
    with _lock:
        cached = _cache.get(key)
    if cached and cached[0] == signature:
        return cached[1]
    start, end = clock(day, root)
    segs = day_player.segments(day, root)
    offset = offset_of(day, root)
    grid = _grid(day, root)
    found = list(_clips(day, root))
    primary = [(day_player._epoch(m['start']), float(m['seconds'])) for p, m in found
               if _on_grid(day_player._epoch(m['start']), grid)]
    measured = day_features.load(day, root)
    tracks, clips, links = {}, [], []
    for path, meta in found:
        state = read_state(path)
        if not state['pieces'] or not (path / ('dets_%s.npz' % TAG)).exists():
            continue
        began = day_player._epoch(meta['start'])
        main = _on_grid(began, grid)
        with np.load(path / ('dets_%s.npz' % TAG)) as archive:
            dets = {cam: archive[cam] for cam in ('cam1', 'cam2')}
        landed = day_player.start_frames(path, meta, segs, day)
        members = {}

        def place(name, cam, rows, fingerprint, floor=None):
            """Rows of one camera become a track on the film's clock, or nothing."""
            rows = np.asarray(rows, np.int64)
            nominal = began + dets[cam][rows, 0]
            if not main:        # an early clip only fills what the grid does not cover
                covered = np.zeros(len(rows), bool)
                for a, length in primary:
                    covered |= (nominal >= a) & (nominal < a + length)
                rows = rows[~covered]
                if not len(rows):
                    return None
            frames = np.rint(dets[cam][rows, 0] * day_player.FPS).astype(np.int64)
            when = day_player.walls(segs[cam], began, frames, landed[cam]) - (offset if cam == 'cam2' else 0.0) - start
            good = ~np.isnan(when)
            if not good.any():
                return None
            order = np.argsort(when[good], kind='stable')
            rows, when = rows[good][order], when[good][order]
            boxes = dets[cam][rows, 1:5].astype(np.float64)
            if floor is None:
                floor = _floor_share(cam, dets[cam][rows])
            seconds = float(when[-1] - when[0])
            tracks[name] = {'key': name, 'clip': path.name, 'cam': cam,
                            'rows': rows, 'times': when, 'first': float(when[0]), 'last': float(when[-1]),
                            'box_first': boxes[0], 'box_last': boxes[-1],
                            'enters_at_edge': _at_edge(boxes[0]), 'leaves_at_edge': _at_edge(boxes[-1]),
                            'shop': bool(floor > 0 or seconds >= SHOP_SECONDS), 'fp': fingerprint}
            return name

        for piece in state['pieces']:
            if not piece['dets']:
                continue
            name = '%s:%d' % (path.name, piece['piece'])
            floor = None
            if measured and name in measured['index']:
                floor = float(measured['floor'][measured['index'][name]])
            if place(name, piece['cam'], piece['dets'], day_player._fingerprint(piece), floor or 0.0
                     if measured and name in measured['index'] else None):
                members[piece['piece']] = name
        # Everything the tracker left out of every piece: 40% of camera 1's detections on 17.09,
        # 9% of camera 2's, two thirds of them confident. The owner saw people with no outline
        # because the film drew pieces only. They are chained into short tracks here.
        for cam in ('cam1', 'cam2'):
            used = {i for p in state['pieces'] if p['cam'] == cam for i in p['dets']}
            for n, chain in enumerate(_leftovers(dets[cam], used)):
                place('%s:x%s.%d' % (path.name, cam[-1], n), cam, chain,
                      'x:%s:%d:%d:%d' % (cam, len(chain), chain[0], chain[-1]))
        groups = read_json(path / ('groups_%s.json' % TAG), None)
        if groups:
            for group in groups:
                names = [members[p] for p in group.get('pieces', []) if p in members]
                links.extend(('group', a, b) for a, b in zip(names, names[1:]))
        owned = [tracks[n] for n in members.values()]
        clips.append({'clip': path.name, 'primary': main, 'nominal': began, 'seconds': float(meta['seconds']), 'landed': landed,
                      'from': round(min((t['first'] for t in owned), default=began - start), 2),
                      'to': round(max((t['last'] for t in owned), default=began - start), 2)})
    clips.sort(key=lambda c: c['nominal'])
    links.extend(_seams(clips, tracks, segs, offset, start))
    result = {'start': start, 'end': end, 'offset': offset, 'tracks': tracks, 'links': links,
              'clips': [{k: c[k] for k in ('clip', 'primary', 'from', 'to')} for c in clips],
              # stretches of film the night passes have labelled at all: elsewhere no outline is
              # not "nobody was there", and the page says so
              'covered': union([[c['nominal'] - start, c['nominal'] - start + c['seconds']] for c in clips])}
    with _lock:
        if key not in _cache and len(_cache) >= 4:
            _cache.pop(next(iter(_cache)))
        _cache[key] = (signature, result)
    return result


def _seams(clips, tracks, segs, offset, start):
    """A track cut by the end of one ten-minute window and the one starting in the next
    window on the same spot a fraction of a second later are one person: the windows run
    back to back and the tracker simply stopped at the border."""
    by_clip = defaultdict(list)
    for t in tracks.values():
        by_clip[t['clip']].append(t)
    main = [c for c in clips if c['primary']]
    out = []
    for a, b in zip(main, main[1:]):
        if abs(b['nominal'] - (a['nominal'] + a['seconds'])) > 2:
            continue
        for cam in ('cam1', 'cam2'):
            border = day_player.walls(segs[cam], b['nominal'], [0], b['landed'][cam])[0] - (offset if cam == 'cam2' else 0.0) - start
            if np.isnan(border):
                continue
            ending = [t for t in by_clip[a['clip']] if t['cam'] == cam and border - SEAM_GAP <= t['last'] <= border + SEAM_GAP]
            starting = [t for t in by_clip[b['clip']] if t['cam'] == cam and border - SEAM_GAP <= t['first'] <= border + SEAM_GAP]
            pairs = sorted(((_iou(x['box_last'], y['box_first']), x['key'], y['key']) for x in ending for y in starting),
                           reverse=True)
            strong = defaultdict(int)
            for score, x, y in pairs:
                if score >= SEAM_IOU:
                    strong[x] += 1
                    strong[y] += 1
            used = set()
            for score, x, y in pairs:
                if score < SEAM_IOU:
                    break
                if x in used or y in used or strong[x] > 1 or strong[y] > 1:
                    continue           # two candidates on one spot: let the owner see it
                used.update((x, y))
                out.append(('seam', x, y))
    return out


# ---------------------------------------------------------------- the owner's corrections

def _store_path(day, root):
    return folder(day, root) / 'edits.json'


def _empty():
    return {'revision': 0, 'undo_revision': None, 'cuts': {}, 'same': [], 'detached': [], 'false': [],
            'kinds': {}, 'missed': [], 'badmask': [], 'prints': {}, 'next_mark': 1}


def store(day, root=ROOT):
    owner = read_json(_store_path(day, root), _empty())
    for key, value in _empty().items():          # files written before a field existed
        owner.setdefault(key, copy.deepcopy(value))
    for pid in owner.pop('staff', []):           # the first release kept staff as a list
        owner['kinds'].setdefault(pid, 'staff')
    return owner


def part_id(track, cut=None):
    return '%s@%s' % (track, '' if cut is None else '%.2f' % cut)


def _parts(info, cuts):
    """[(part id, first, last)] of one track after the owner's cuts."""
    edges = [c for c in sorted(cuts) if info['first'] < c < info['last']]
    bounds = [None] + edges
    out = []
    for n, cut in enumerate(bounds):
        lo = info['first'] if cut is None else cut
        hi = edges[n] if n < len(edges) else info['last']
        inside = info['times'][(info['times'] >= lo - 1e-6) & (info['times'] <= hi + 1e-6)]
        out.append((part_id(info['key'], cut), float(inside[0]) if len(inside) else lo,
                    float(inside[-1]) if len(inside) else hi))
    return out


def _stale(owner, tracks):
    return {k for k, fp in owner.get('prints', {}).items() if k in tracks and tracks[k]['fp'] != fp}


def _all(day, root=ROOT):
    """The machine's tracks and the people the owner drew himself (day_masks), with the joins
    the carrying made when an ordinary track picked a drawn person up."""
    import day_masks
    data = index(day, root)
    extra, joins = day_masks.tracks(day, root)
    if not extra:
        return data
    merged = dict(data)
    merged['tracks'] = {**data['tracks'], **extra}
    merged['links'] = data['links'] + joins
    return merged


def people(day, root=ROOT, owner=None):
    """The day's people: the machine's links, minus what the owner took apart, plus what he
    joined. Recomputed from scratch on every edit -- a few thousand parts, milliseconds.

    Numbers and colours stay put while he works: a person keeps the number the machine's
    own grouping gave the earliest of his pieces, and whoever an edit creates gets a new
    number after the last one. The owner remembers "#57"; an X on #12 must not renumber it."""
    machine = index(day, root)
    data = _all(day, root)
    owner = owner or store(day, root)
    crowd = _components(data, owner)
    # numbers come from the machine alone: a drawn person is new and numbered after the rest
    base = crowd if owner == _empty() and data is machine else _components(machine, _empty())
    by_track = {}
    for pid, part in base['parts'].items():
        by_track[part['track']] = base['persons'][part['person']].get('order')
    # passers-by behind the glass are outlined and corrected like anybody, but not numbered:
    # there are two thousand of them a day, and a number is only there to be remembered
    shown = sorted((a for a, v in crowd['persons'].items() if v['kind'] != 'passer'),
                   key=lambda a: (crowd['persons'][a]['first'], a))
    used, later = set(), []
    for anchor in shown:
        person = crowd['persons'][anchor]
        # only the piece a track starts with carries the machine's number: the half the owner cut
        # off is somebody new and keeps a number of its own, whatever happens to the first half
        mine = sorted({by_track.get(crowd['parts'][p]['track']) for p in person['members'] if p.endswith('@')} - {None})
        free = [n for n in mine if n not in used]
        if free:
            person['n'] = free[0]
            used.add(free[0])
        else:
            later.append(anchor)
    top = max([p.get('order') or 0 for p in base['persons'].values()] + list(used) + [0])
    for step, anchor in enumerate(later, 1):
        crowd['persons'][anchor]['n'] = top + step
    busy = []                               # (first, last, colour) of people already coloured
    for anchor in sorted(shown, key=lambda a: crowd['persons'][a]['n']):
        person = crowd['persons'][anchor]
        taken = {c for first, last, c in busy if first - 20 <= person['last'] and last + 20 >= person['first']}
        wish = (person['n'] - 1) % 12
        person['color'] = wish if wish not in taken else next((c for c in range(12) if c not in taken), wish)
        busy.append((person['first'], person['last'], person['color']))
    for person in crowd['persons'].values():
        person.pop('members', None)
        person.pop('order', None)
        person.setdefault('n', None)
        person.setdefault('color', None)
    return crowd


def _components(data, owner):
    tracks = data['tracks']
    stale = _stale(owner, tracks)
    parts, head, tail = {}, {}, {}
    for key, info in tracks.items():
        cuts = [] if key in stale else owner['cuts'].get(key, [])
        pieces = _parts(info, cuts)
        for pid, first, last in pieces:
            parts[pid] = {'track': key, 'cam': info['cam'], 'first': round(first, 2), 'last': round(last, 2),
                          'shop': info['shop']}
        head[key], tail[key] = pieces[0][0], pieces[-1][0]
    def live(pid):
        return pid in parts and parts[pid]['track'] not in stale
    false = {p for p in owner['false'] if live(p)}
    detached = {p for p in owner['detached'] if live(p)}
    parent = {p: p for p in parts if p not in false}

    def find(p):
        while parent[p] != p:
            parent[p] = parent[parent[p]]
            p = parent[p]
        return p

    def join(a, b):
        if a in parent and b in parent:
            ra, rb = find(a), find(b)
            if ra != rb:
                if (parts[ra]['first'], ra) < (parts[rb]['first'], rb):
                    parent[rb] = ra
                else:
                    parent[ra] = rb
    for kind, a, b in data['links']:
        if kind == 'join':                  # a drawn person carried onto an ordinary track's piece
            pa, pb = head.get(a), b
        elif kind == 'group':
            pa, pb = head.get(a), head.get(b)
        else:
            pa, pb = tail.get(a), head.get(b)
        if pa and pb and pa not in detached and pb not in detached:
            join(pa, pb)
    for a, b in owner['same']:
        if live(a) and live(b):
            join(a, b)
    members = defaultdict(list)
    for p in parent:
        members[find(p)].append(p)
    marks = {p: k for p, k in owner.get('kinds', {}).items() if live(p) and k in KINDS}
    persons = {}
    for anchor, group in members.items():
        shop = any(parts[p]['shop'] for p in group)
        mine = set(group)
        said = [k for p, k in marks.items() if p in mine]           # the owner's latest word wins
        kind = said[-1] if said else ('customer' if shop else 'passer')
        persons[anchor] = {'first': min(parts[p]['first'] for p in group),
                           'last': max(parts[p]['last'] for p in group), 'shop': shop, 'kind': kind,
                           'staff': kind == 'staff', 'parts': len(group), 'members': group}
    shown = sorted((a for a, v in persons.items() if v['shop']), key=lambda a: (persons[a]['first'], a))
    for n, anchor in enumerate(shown, 1):
        persons[anchor]['order'] = n            # the machine's own numbering, by first appearance
    for pid, part in parts.items():
        part['person'] = None if pid in false else find(pid)
        part['false'] = pid in false
    return {'parts': parts, 'persons': persons, 'stale': sorted(stale), 'detached': sorted(detached)}


def doubts(day, root=ROOT, crowd=None):
    """Moments to slow down at: a visitor's part began or ended in the middle of the room,
    and nothing the machine knows carries it on."""
    data = _all(day, root)
    crowd = crowd or people(day, root)
    # Only a seam link is sure (none ambiguous on 17.09). A track the grouping joined to
    # somebody is exactly where the grouping goes wrong (15% of its groups mix two people).
    linked_in = {b for kind, a, b in data['links'] if kind == 'seam'}
    linked_out = {a for kind, a, b in data['links'] if kind == 'seam'}
    moments = []
    for pid, part in crowd['parts'].items():
        if not part['shop'] or part['false']:
            continue
        info = data['tracks'][part['track']]
        if info.get('drawn'):               # the owner made it; nothing to doubt
            continue
        whole_start = pid.endswith('@')
        if whole_start and not info['enters_at_edge'] and part['track'] not in linked_in:
            moments.append(part['first'])
        if abs(part['last'] - info['last']) < 0.01 and not info['leaves_at_edge'] and part['track'] not in linked_out:
            moments.append(part['last'])
    owner = store(day, root)
    for moment, track, score in read_json(_switches_path(day, root), {'items': []})['items']:
        cut_here = any(abs(c - moment) < 2 for c in owner['cuts'].get(track, []))
        if score >= SWITCH and track in data['tracks'] and not cut_here:
            moments.append(moment)
    moments.sort()
    merged = []
    for m in moments:
        if not merged or m - merged[-1] > 3:
            merged.append(m)
    return [round(m, 2) for m in merged]


SWITCH = 0.25               # look change between the second before and the second after
SWITCH_WINDOW = 8           # detections either side: about a second at 8.3 looks a second


def _switches_path(day, root):
    return folder(day, root) / 'switches.json'


def switches(day, root=ROOT, log=print):
    """Moments inside one track where the person's look jumps: the tracker may have handed
    the track to somebody else, and the colour on screen will not change to show it. Found
    once, from the descriptors the night pass already stored; the page slows down there.
    Every peak above 0.15 is kept with its score, so the threshold can be moved without
    running this again."""
    root = Path(root)
    data = index(day, root)
    by_clip = defaultdict(list)
    for info in data['tracks'].values():
        if info['shop']:
            by_clip[info['clip']].append(info)
    found, w = [], SWITCH_WINDOW
    for clip, items in sorted(by_clip.items()):
        path = root / 'data/raw_clips' / clip / EMB
        if not path.exists():
            continue
        with np.load(path) as archive:
            looks = {cam: archive[cam] for cam in ('cam1', 'cam2')}
        for info in items:
            rows, every = info['rows'], looks[info['cam']]
            if len(rows) < 3 * w or rows.max() >= len(every):
                continue
            v = every[rows].astype(np.float64)
            v /= np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-8)
            total = np.vstack([np.zeros((1, v.shape[1])), np.cumsum(v, axis=0)])
            at = np.arange(w, len(rows) - w + 1)
            before, after = total[at] - total[at - w], total[at + w] - total[at]
            cosine = (before * after).sum(1) / np.maximum(
                np.linalg.norm(before, axis=1) * np.linalg.norm(after, axis=1), 1e-8)
            jump = 1 - cosine
            kept = []
            for j in np.argsort(-jump):
                if jump[j] < 0.15:
                    break
                moment = float(info['times'][at[j] - 1] + info['times'][at[j]]) / 2
                if any(abs(moment - k) < 2 for k in kept):
                    continue
                kept.append(moment)
                found.append([round(moment, 2), info['key'], round(float(jump[j]), 3)])
        log('%s: %d looks checked' % (clip, len(items)))
    found.sort()
    atomic_json(_switches_path(day, root), {'version': VERSION, 'items': found})
    log(json.dumps({'day': day, 'peaks': len(found), 'above': sum(1 for f in found if f[2] >= SWITCH)}))
    return found


def activity(crowd, margin=2.0):
    spans = sorted((p['first'] - margin, p['last'] + margin) for p in crowd['parts'].values()
                   if not p['false'])
    out = []
    for a, b in spans:
        if out and a <= out[-1][1]:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return [[round(max(a, 0), 2), round(b, 2)] for a, b in out]


def _change_point(day, info, at, rate, root):
    """Where inside the last few seconds the look of the track jumps the most. The owner
    pressed S while the film ran at `rate`, so his moment is late by about half a second of
    his time; the embeddings say where the other person actually took over."""
    lo = at - max(3.0, 1.2 * rate)
    fallback = max(info['first'], at - 0.5 * rate)
    path = Path(root) / 'data/raw_clips' / info['clip'] / EMB
    if not path.exists():
        return fallback
    pick = (info['times'] >= lo) & (info['times'] <= at + 0.5)
    rows, times = info['rows'][pick], info['times'][pick]
    if len(rows) < 6:
        return fallback
    with np.load(path) as archive:
        looks = archive[info['cam']]
    if rows.max() >= len(looks):
        return fallback
    looks = looks[rows].astype(np.float64)
    looks /= np.maximum(np.linalg.norm(looks, axis=1, keepdims=True), 1e-8)
    best, where = 0.0, None
    for j in range(3, len(rows) - 2):
        a, b = looks[:j].mean(0), looks[j:].mean(0)
        distance = 1 - float(a @ b) / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-8)
        if distance > best:
            best, where = distance, (times[j - 1] + times[j]) / 2
    return where if where is not None and best >= CHANGE else fallback


def change(day, body, root=ROOT):
    """One correction. Every edit names the revision it was made against; the previous
    state goes to history first, so undo always has something to return to."""
    root = Path(root)
    path = _store_path(day, root)
    action = body.get('action')
    with file_lock(path.with_suffix('.lock')):
        current = store(day, root)
        if body.get('revision') != current['revision']:
            raise ValueError('Разметка изменилась в другой вкладке; обновите страницу')
        history = path.parent / 'history' / ('%08d.json' % current['revision'])
        before = copy.deepcopy(current)
        if action == 'undo':
            back = current.get('undo_revision')
            previous = read_json(path.parent / 'history' / ('%08d.json' % back)) if back is not None else None
            if previous is None:
                raise ValueError('Отменять нечего')
            atomic_json(history, current)
            previous['revision'] = current['revision'] + 1
            previous['undo_revision'] = None
            atomic_json(path, previous)
            return {'revision': previous['revision'], 'did': 'undo'}
        data = _all(day, root)
        crowd = people(day, root, current)
        parts = crowd['parts']

        def known(pid):
            if pid not in parts:
                raise ValueError('Этого куска больше нет; обновите страницу')
            track = parts[pid]['track']
            current['prints'][track] = data['tracks'][track]['fp']
            return parts[pid]

        did = {}
        if action == 'split':
            pid, at = body.get('part'), body.get('at')
            part = known(pid)
            if not isinstance(at, (int, float)):
                raise ValueError('Не указан момент')
            info = data['tracks'][part['track']]
            rate = float(body.get('rate') or 1)
            cut = _change_point(day, info, float(at), rate, root) if rate > 1.5 else float(at)
            cut = round(min(max(cut, part['first']), part['last']), 2)
            after = info['times'][(info['times'] > cut) & (info['times'] <= part['last'] + 1e-6)]
            if cut - part['first'] >= 1.0 and not len(after):
                raise ValueError('Здесь этот кусок уже кончился')
            if cut - part['first'] < 1.0:
                target = pid                  # the whole piece belongs to somebody else
            else:
                current['cuts'].setdefault(part['track'], []).append(cut)
                current['cuts'][part['track']].sort()
                target = part_id(part['track'], cut)
            if target not in current['detached']:
                current['detached'].append(target)
            did = {'did': 'split', 'part': target, 'at': part['first'] if target == pid else cut}
        elif action == 'same':
            pid, other = body.get('part'), body.get('target')
            a, b = known(pid), known(other)
            if a['person'] is not None and a['person'] == b['person']:
                raise ValueError('Это уже один человек')
            for x, y in ((pid, other), (other, pid)):
                if x in current['false']:
                    current['false'].remove(x)
            current['same'].append([pid, other])
            did = {'did': 'same', 'part': pid, 'target': other}
        elif action == 'false':
            pid = body.get('part')
            known(pid)
            if pid in current['false']:
                current['false'].remove(pid)
            else:
                current['false'].append(pid)
            did = {'did': 'false', 'part': pid, 'now': pid in current['false']}
        elif action in ('kind', 'staff'):
            # who somebody is: a customer, staff, or a passer-by behind the glass. Said once, it
            # covers the whole person; the earlier word about any of his pieces is replaced.
            pid = body.get('part')
            person = known(pid)['person']
            if person is None:
                raise ValueError('Это отмечено как не человек')
            kind = body.get('kind')
            if action == 'staff':               # the first release: a toggle
                kind = 'customer' if crowd['persons'][person]['kind'] == 'staff' else 'staff'
            if kind not in KINDS:
                raise ValueError('Неизвестно, кто это')
            mine = [p for p in current['kinds'] if p in parts and parts[p]['person'] == person]
            for p in mine:
                del current['kinds'][p]
            current['kinds'][pid] = kind
            did = {'did': 'kind', 'part': pid, 'kind': kind, 'now': kind == 'staff'}
        elif action == 'badmask':
            # the outline is wrong here (half a person, two people, a piece of the stand): these
            # frames must not teach the student, and the night can outline them again
            pid, at = body.get('part'), body.get('at')
            part = known(pid)
            if not isinstance(at, (int, float)):
                raise ValueError('Не указан момент')
            near = [m for m in current['badmask'] if m['part'] == pid and abs(m['at'] - at) < 1.5]
            if near:
                current['badmask'] = [m for m in current['badmask'] if m not in near]
                did = {'did': 'badmask', 'now': False}
            else:
                current['badmask'].append({'id': current['next_mark'], 'part': pid, 'cam': part['cam'],
                                           'at': round(float(at), 2)})
                current['next_mark'] += 1
                did = {'did': 'badmask', 'now': True}
        elif action == 'missed':
            # x, y in pixels of the original 2560x1440 frame
            cam, at = body.get('cam'), body.get('at')
            x, y = body.get('x'), body.get('y')
            if cam not in ('cam1', 'cam2') or not all(isinstance(v, (int, float)) for v in (at, x, y)):
                raise ValueError('Не указано место')
            near = [m for m in current['missed'] if m['cam'] == cam and abs(m['at'] - at) < 1.5
                    and abs(m['x'] - x) < 100 and abs(m['y'] - y) < 160]
            if near:                           # a second M on the same spot takes the mark back
                current['missed'] = [m for m in current['missed'] if m not in near]
                did = {'did': 'missed', 'now': False}
            else:
                mark = {'id': current['next_mark'], 'cam': cam, 'at': round(float(at), 2),
                        'x': round(float(x), 1), 'y': round(float(y), 1), 'units': 'raw'}
                current['next_mark'] += 1
                current['missed'].append(mark)
                did = {'did': 'missed', 'now': True}
        else:
            raise ValueError('Неизвестное действие')
        atomic_json(history, before)
        current['undo_revision'] = current['revision']
        current['revision'] += 1
        atomic_json(path, current)
    did['revision'] = current['revision']
    return did


# ---------------------------------------------------------------- what was watched

def _watched_path(day, root):
    return folder(day, root) / 'watched.json'


def union(spans):
    out = []
    for a, b in sorted((float(a), float(b)) for a, b, *_ in spans):
        if out and a <= out[-1][1] + 0.2:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return out


def watched(day, root=ROOT):
    record = read_json(_watched_path(day, root), {'spans': []})
    merged = union(record['spans'])
    return {'spans': [[round(a, 2), round(b, 2)] for a, b in merged],
            'seconds': round(sum(b - a for a, b in merged), 1),
            'fastest': max((s[2] for s in record['spans'] if len(s) > 2), default=None)}


def add_watched(day, spans, root=ROOT):
    """Stretches of film the owner watched play, with the speed. Kept raw: the speed is part
    of what "watched" means, and a later reader may want to count only the slow ones."""
    start, end = clock(day, root)
    clean = []
    for item in spans if isinstance(spans, list) else []:
        if not isinstance(item, list) or len(item) < 3:
            continue
        a, b, rate = (float(v) for v in item[:3])
        if 0 <= a < b <= end - start + 1 and b - a <= 300 and 0 < rate <= 32:
            clean.append([round(a, 2), round(b, 2), round(rate, 2)])
    if not clean:
        return watched(day, root)
    path = _watched_path(day, root)
    with file_lock(path.with_suffix('.lock')):
        record = read_json(path, {'spans': []})
        record['spans'].extend(clean)
        atomic_json(path, record)
    return watched(day, root)


# ---------------------------------------------------------------- what the page loads

def overview(day, root=ROOT):
    import day_masks
    root = Path(root)
    start, end = clock(day, root)
    data = index(day, root)
    owner = store(day, root)
    crowd = people(day, root, owner)
    offset = offset_of(day, root)
    videos = {}
    for cam in ('cam1', 'cam2'):
        mp4, side = film(day, cam, root)
        record = read_json(side, None)
        ready = bool(record and mp4.exists() and record.get('start') == start and record.get('version') == VERSION)
        # a film cut with another camera offset shows camera 2 at the wrong moment
        stale = bool(ready and cam == 'cam2' and abs(float(record.get('shift', 0)) - offset) > 0.02)
        videos[cam] = {'ready': ready and not stale, 'stale': stale, 'url': '/movie-file/%s/%s.mp4' % (day, cam),
                       'holes': record.get('holes', []) if record else [],
                       'size': mp4.stat().st_size if ready else 0}
    # [[moment, part id], ...] per cut track: the page looks the part up, never formats it
    cuts = {k: [[c, part_id(k, c)] for c in sorted(v)] for k, v in owner['cuts'].items()
            if k not in crowd['stale'] and k in data['tracks']}
    return {'day': day, 'start': start, 'duration': round(end - start, 2), 'tick': TICK,
            'width': WIDTH, 'height': HEIGHT, 'offset': data['offset'], 'videos': videos,
            'clips': data['clips'], 'covered': [[round(a, 2), round(b, 2)] for a, b in data['covered']], 'cuts': cuts,
            'parts': {pid: [p['person'], p['first'], p['last'], int(p['shop']), int(p['false'])]
                      for pid, p in crowd['parts'].items()},
            'persons': {a: {k: p[k] for k in ('n', 'color', 'kind', 'first', 'last', 'parts')}
                        for a, p in crowd['persons'].items()},
            'doubts': doubts(day, root, crowd), 'activity': activity(crowd),
            'missed': owner['missed'], 'badmask': owner['badmask'], 'unit': UNIT,
            'drawn': day_masks.view(day, root, UNIT), 'segmenter': Path(day_masks._weights()).name,
            'watched': watched(day, root), 'revision': owner['revision'],
            'can_undo': owner.get('undo_revision') is not None, 'stale': crowd['stale'],
            'edits': {'same': len(owner['same']), 'detached': len(owner['detached']),
                      'false': len(owner['false']), 'kinds': len(owner['kinds']), 'missed': len(owner['missed']),
                      'badmask': len(owner['badmask'])}}


_KERNEL = np.ones((3, 3), np.uint8)


def clean_outline(outline, unit=UNIT, eps=0.8):
    """Rings of one mask in `unit`-wide pixels, as a person looks and not as it was stored.

    The detector saved each mask as one contour: Ultralytics joins the pieces of a mask that
    fell apart (head and body split by a stand) with zero-width bridges, and the contour was
    then thinned to about 64 points by taking every n-th one -- which turns every bridge into
    a spike across the person (about 1% of outlines have an edge longer than half the body).
    Drawn as stored, and thinned again for the page, masks looked crooked. Here the outline is
    filled, opened by one pixel (bridges and spikes go, the body stays), traced again, and
    simplified by shape, so the head and the feet keep their points.
    """
    import cv2
    k = unit / 2560.0
    p = np.asarray(outline, np.float64).reshape(-1, 2) * k
    if len(p) < 3:
        return []
    x0, y0 = np.floor(p.min(0)) - 3
    x1, y1 = np.ceil(p.max(0)) + 3
    if (x1 - x0) * (y1 - y0) > 4e6:
        return [np.rint(p).astype(int)]
    filled = np.zeros((int(y1 - y0) + 1, int(x1 - x0) + 1), np.uint8)
    cv2.fillPoly(filled, [np.rint((p - [x0, y0]) * 4).astype(np.int32)], 1, shift=2)
    opened = cv2.morphologyEx(filled, cv2.MORPH_OPEN, _KERNEL)
    if opened.sum() < 0.3 * filled.sum():      # a tiny far figure: opening would erase it
        opened = filled
    contours, _ = cv2.findContours(opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    rings = []
    for c in contours:
        if cv2.contourArea(c) < 6:
            continue
        ring = cv2.approxPolyDP(c, eps, True).reshape(-1, 2) + [x0, y0]
        if len(ring) >= 3:
            rings.append(np.rint(ring).astype(int))
    return rings or [np.rint(p).astype(int)]


def clip_marks(day, clip, root=ROOT):
    """Every outline of one clip that the film shows, on the film's clock, gzip-packed JSON.
    Cleaning a busy window takes a couple of seconds, so the result is kept on disk until the
    clip, the day's layout or the cleaning itself changes."""
    import hashlib
    root = Path(root)
    data = index(day, root)
    base = root / 'data/raw_clips' / clip
    polys = base / ('polys_%s.npz' % TAG)
    stamp = hashlib.sha1(json.dumps([_signature(day, root), OUTLINE_VERSION, UNIT, clip],
                                    default=str).encode()).hexdigest()[:12]
    with _lock:
        if stamp in _clip_cache:
            return _clip_cache[stamp]
    kept = folder(day, root) / 'outlines' / ('%s_%s.json.gz' % (clip, stamp))
    if kept.exists():
        body = kept.read_bytes()
    else:
        shapes = None
        if polys.exists():
            with np.load(polys) as archive:
                shapes = {cam: (archive[cam + '_pts'], archive[cam + '_off']) for cam in ('cam1', 'cam2')}
        with np.load(base / ('dets_%s.npz' % TAG)) as archive:
            dets = {cam: archive[cam] for cam in ('cam1', 'cam2')}
        out = []
        for key, info in data['tracks'].items():
            if info['clip'] != clip:
                continue
            cam = info['cam']
            outlines = []
            for row in info['rows']:
                outline = None
                if shapes is not None and row + 1 < len(shapes[cam][1]):
                    points, offsets = shapes[cam]
                    outline = points[offsets[row]:offsets[row + 1]]
                if outline is None or len(outline) < 3:
                    x1, y1, x2, y2 = dets[cam][row, 1:5]
                    outline = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], np.float32)
                outlines.append([ring.ravel().tolist() for ring in clean_outline(outline)])
            out.append({'k': key, 'cam': 1 if cam == 'cam1' else 2,
                        't': [round(float(t), 2) for t in info['times']], 'p': outlines})
        body = gzip.compress(json.dumps({'clip': clip, 'unit': UNIT, 'tracks': out}, separators=(',', ':')).encode(), 6)
        kept.parent.mkdir(parents=True, exist_ok=True)
        for old in kept.parent.glob('%s_*.json.gz' % clip):
            old.unlink(missing_ok=True)
        partial = kept.with_suffix('.part')
        partial.write_bytes(body)
        os.replace(partial, kept)
    with _lock:
        if len(_clip_cache) >= 12:
            _clip_cache.pop(next(iter(_clip_cache)))
        _clip_cache[stamp] = body
    return body


def days(root=ROOT):
    out = []
    for side in sorted((Path(root) / 'data/day_movie').glob('*/cam1.json')):
        out.append(side.parent.name)
    return out


def register(app, root):
    """`root` is a callable, read on every request like the other routes of the app."""
    import re
    from flask import Response, abort, jsonify, render_template, request, send_file
    here = root if callable(root) else (lambda: root)

    def valid_day(day):
        if not re.fullmatch(r'\d{8}', day):
            abort(400)

    def packed(value):
        body = json.dumps(value, separators=(',', ':'), ensure_ascii=False).encode()
        if 'gzip' in request.headers.get('Accept-Encoding', '') and len(body) > 2048:
            response = Response(gzip.compress(body, 5), mimetype='application/json')
            response.headers['Content-Encoding'] = 'gzip'
            return response
        return Response(body, mimetype='application/json')

    @app.get('/movie')
    def movie_page():
        return render_template('movie.html')

    @app.get('/api/movie')
    def movie_days():
        return jsonify({'days': days(Path(here()))})

    @app.get('/api/movie/<day>')
    def movie_overview(day):
        valid_day(day)
        try:
            return packed(overview(day, Path(here())))
        except (ValueError, KeyError, OSError) as exc:
            return jsonify({'error': str(exc)}), 400

    @app.get('/api/movie/<day>/clip/<clip>')
    def movie_clip(day, clip):
        valid_day(day)
        if not re.fullmatch(r'c[A-Za-z0-9_-]+', clip):
            abort(400)
        if not (Path(here()) / 'data/raw_clips' / clip / ('dets_%s.npz' % TAG)).exists():
            abort(404)
        body = clip_marks(day, clip, Path(here()))
        response = Response(body, mimetype='application/json')
        response.headers['Content-Encoding'] = 'gzip'
        response.headers['Cache-Control'] = 'no-cache'
        return response

    @app.post('/api/movie/<day>')
    def movie_change(day):
        valid_day(day)
        try:
            did = change(day, request.get_json(force=True), Path(here()))
        except (ValueError, KeyError, TypeError) as exc:
            return jsonify({'error': str(exc)}), 409
        view = overview(day, Path(here()))
        view['did'] = did
        return packed(view)

    @app.post('/api/movie/<day>/watched')
    def movie_watched(day):
        valid_day(day)
        body = request.get_json(force=True) or {}
        return jsonify(add_watched(day, body.get('spans'), Path(here())))

    @app.get('/movie-file/<day>/<cam>.mp4')
    def movie_file(day, cam):
        valid_day(day)
        if cam not in ('cam1', 'cam2'):
            abort(404)
        path, _ = film(day, cam, Path(here()))
        if not path.exists():
            abort(404)
        response = send_file(path, conditional=True, mimetype='video/mp4')
        response.headers['Cache-Control'] = 'private, max-age=3600'
        return response

    import day_masks
    day_masks.register(app, root)


def prepare(day, root=ROOT, log=print):
    """Everything a day needs to be watched, CPU only: the light copies that are missing
    (day_proxy skips the ones already made), both films, the look jumps. The films do not
    wait for the night passes: outlines of windows labelled later appear on their own."""
    began = time.time()
    subprocess.run([sys.executable, str(Path(__file__).with_name('day_proxy.py')), day], cwd=str(ROOT), check=False)
    log('light copies ready in %.0f s' % (time.time() - began))
    starts(day, root, log=log)
    for cam in ('cam1', 'cam2'):
        build(day, cam, root, log=log)
    switches(day, root, log=log)
    outlines(day, root, log=log)
    log('day %s ready to watch in %.0f s' % (day, time.time() - began))


def starts(day, root=ROOT, log=print):
    """Replay, once per clip and camera, which frame the detector really started on
    (day_player.start_frames), while the raw recordings still exist."""
    began, segs, shifted = time.time(), day_player.segments(day, root), 0
    for path, meta in _clips(day, root):
        landed = day_player.start_frames(path, meta, segs, day)
        record = read_json(path / 'start_frames.json', {})
        shifted += sum(1 for cam in record.values() if abs(cam.get('shift_s', 0)) > 0.5)
    log('start frames of %s: %d camera-clips off by more than 0.5 s, %.0f s' % (day, shifted, time.time() - began))


def outlines(day, root=ROOT, log=print):
    """Clean every window's outlines ahead of the owner: a busy window takes ~8 s the first
    time (measured on 17.09 17:10), which he should not wait for when the film gets there."""
    began = time.time()
    for item in index(day, root)['clips']:
        if (Path(root) / 'data/raw_clips' / item['clip'] / ('dets_%s.npz' % TAG)).exists():
            clip_marks(day, item['clip'], root)
    log('outlines of %s ready in %.0f s' % (day, time.time() - began))


if __name__ == '__main__':
    say = lambda s: print(time.strftime('%H:%M:%S'), s, flush=True)
    if len(sys.argv) == 3 and sys.argv[1] == 'prepare':
        _below_normal()
        prepare(sys.argv[2], log=say)
    elif len(sys.argv) >= 3 and sys.argv[1] == 'build':
        _below_normal()
        for cam in (sys.argv[3:] or ['cam1', 'cam2']):
            build(sys.argv[2], cam, log=say)
        switches(sys.argv[2], log=say)
        outlines(sys.argv[2], log=say)
    elif len(sys.argv) == 3 and sys.argv[1] == 'outlines':
        _below_normal()
        starts(sys.argv[2], log=say)
        outlines(sys.argv[2], log=say)
    elif len(sys.argv) == 3 and sys.argv[1] == 'switches':
        _below_normal()
        switches(sys.argv[2], log=say)
    else:
        sys.exit('usage: day_movie.py prepare DAY | build DAY [cam1|cam2] | switches DAY | outlines DAY')
