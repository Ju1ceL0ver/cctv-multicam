"""Bounded CPU replay of TRAIN predictions through the real side/view queues.

No inference, training, live files or owner labels are changed. Model role
service is deliberately excluded: this measures side and view selection only.
"""
import argparse
import hashlib
import json
import pickle
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
TAGS = ('door_20260918_01405', 'door_20260918_15375')


def run(out, ticks, fast=False, fresh=False, start_tick=0):
    if fast:
        from door_seg_queue_fast import Live
    else:
        from door_combo import Live
    from door_role import Bank, box_iou, score, ISOLATED_IOU
    from door_seg_track import Tracker
    from staff_masked import crop_masked
    cv2.setNumThreads(2)
    out.mkdir(parents=True, exist_ok=False)
    report = {'complete': False, 'fast': fast, 'fresh_reference': fresh, 'sources': {}, 'limitations':
              'CPU cached TRAIN replay; excludes GPU segmentation, CRM, live scheduling and role model service.'}
    for tag in TAGS:
        path = ROOT / 'data/quality_backups/doorseg_quick' / (tag + '.pkl')
        data = pickle.loads(path.read_bytes())
        current = json.loads((ROOT / 'data/door_side/20260918.json').read_text(encoding='utf-8'))[tag]
        if not all(s.get('done') and s.get('role') == 'train' for s in (data['state'], current)):
            raise ValueError('completed owner TRAIN required: ' + tag)
        combo, tracker = Live(), Tracker()
        bank = Bank(out / tag / 'views')
        cap = cv2.VideoCapture(str(ROOT / 'data/sam31_door' / tag / 'cam1/video.mp4'))
        if start_tick and not cap.set(cv2.CAP_PROP_POS_FRAMES,start_tick):
            cap.release()
            raise ValueError('could not seek TRAIN excerpt')
        records = data['records'][start_tick:start_tick+ticks]
        if len(records) != ticks:
            cap.release()
            raise ValueError('not enough cached TRAIN ticks')
        waits, service, ids, vectors = [], [], [], {}
        resized = []
        original_small = combo.clf._small
        if fresh:
            def original_small(frame):
                rgb = cv2.resize(frame, (1280, 720), interpolation=cv2.INTER_LINEAR)
                return rgb, cv2.resize(rgb.astype(np.float32), (320,180), interpolation=cv2.INTER_AREA).astype(np.uint8)

        def small(frame):
            key = (id(frame), frame.shape, frame.ctypes.data)
            c = getattr(combo.clf, '_frame_cache', None)
            resized.append(c is None or c[0] != key)
            return original_small(frame)

        combo.clf._small = small
        pool = combo._pool()
        submit = pool.submit

        def timed_submit(fn):
            queued = time.perf_counter()
            index = len(combo.pending)
            def call():
                start = time.perf_counter()
                waits.append(start - queued)
                try:
                    value = fn()
                    vectors[index] = value
                    return value
                finally:
                    service.append(time.perf_counter() - start)
            return submit(call)

        pool.submit = timed_submit
        decode_s = bank_s = add_s = 0.
        started = time.perf_counter()
        person_of = {}
        try:
            for index, (tick, packed, _) in enumerate(records):
                if tick != start_tick+index:
                    raise ValueError('probe requires contiguous TRAIN cache')
                if time.perf_counter() - started > 90:
                    raise TimeoutError('bounded CPU probe expired')
                t = time.perf_counter()
                ok, frame = cap.read()
                if not ok:
                    raise ValueError('short TRAIN video')
                frame = cv2.resize(frame, (1280, 720), interpolation=cv2.INTER_AREA)[:, :, ::-1].copy()
                decode_s += time.perf_counter() - t
                pred = [(np.unpackbits(z[0], count=184*320).reshape(184, 320).astype(bool), *z[1:])
                        for z in packed if z[2] >= .4]
                linked = tracker.update(tick, pred)
                boxes = [np.asarray(p[1], float) for p in pred]
                for row, (pid, p) in enumerate(linked):
                    t = time.perf_counter()
                    mask = cv2.resize(p[0][:180].astype(np.uint8), (1280, 720), interpolation=cv2.INTER_NEAREST)
                    yy, xx = np.nonzero(mask)
                    if not len(xx):
                        continue
                    x1, y1, x2, y2 = int(xx.min()), int(yy.min()), int(xx.max()+1), int(yy.max()+1)
                    crop = mask[y1:y2, x1:x2]
                    bottom = yy.max()
                    foot = [float(xx[yy >= bottom-2].mean())/1280, float(bottom)/720]
                    key = (tag, tick, row)
                    person_of[key] = str(pid)
                    combo.add(key, tick*.08, frame, (crop, x1, y1), foot)
                    add_s += time.perf_counter() - t
                    t = time.perf_counter()
                    bx = boxes[row]
                    isolated = all(box_iou(bx, other) <= ISOLATED_IOU for j, other in enumerate(boxes) if j != row)
                    cut = x1 <= 2 or y1 <= 2 or x2 >= 1277 or y2 >= 717
                    sc = score(int(crop.sum()*(2176/1280)**2), isolated, cut)
                    bank.offer(str(pid), sc, lambda: crop_masked(frame[:, :, ::-1], mask > 0, [x1,y1,x2,y2]), [x1,y1,x2,y2])
                    bank_s += time.perf_counter() - t
                    ids.append([tick, row, pid])
            pending = len(combo.pending)
            t = time.perf_counter()
            # Nonempty crossings contain numpy coordinates; normalize only the
            # saved report representation, leaving the actual event policy intact.
            events = json.loads(json.dumps(combo.events(None, person_of),
                                          default=lambda value: value.tolist()))
            drain = time.perf_counter() - t
        finally:
            cap.release()
            pool.shutdown(wait=True)
        bank.dump()
        summary = lambda a: {'mean_ms': float(np.mean(a)*1000) if a else 0.,
                            'p90_ms': float(np.quantile(a,.9)*1000) if a else 0.,
                            'max_ms': float(max(a)*1000) if a else 0.}
        report['sources'][tag] = {'ticks': ticks, 'start_tick': start_tick, 'rows': len(ids), 'feature_jobs': len(service),
            'small_calls': len(resized), 'small_cache_misses': sum(resized),
            'feature_wait': summary(waits), 'feature_service': summary(service),
            'decode_seconds': decode_s, 'add_seconds': add_s, 'bank_seconds': bank_s,
            'pending_before_drain': pending, 'drain_seconds': drain,
            'total_seconds': time.perf_counter()-started, 'kept_views': sum(map(len, bank.views.values())),
            'offered_views': bank.n, 'events': events, 'tracker_stats': tracker.stats,
            'feature_sha256': hashlib.sha256(np.stack([vectors[i] for i in sorted(vectors)]).tobytes()).hexdigest() if vectors else None,
            'votes_sha256': hashlib.sha256(json.dumps(list(combo.obs.items())).encode()).hexdigest(),
            'view_sha256': hashlib.sha256(b''.join(p.read_bytes() for p in sorted(bank.folder.glob('*.jpg'), key=lambda p: int(p.stem)))).hexdigest(),
            'ids_sha256': hashlib.sha256(json.dumps(ids).encode()).hexdigest(),
            'cache_sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
        if fast:
            report['sources'][tag]['prepared_cache'] = {'hits': combo.prepared_frames.hits,
                'misses': combo.prepared_frames.misses, 'retained': len(combo.prepared_frames.frames)}
        np.savez_compressed(bank.folder.parent/'features.npz', features=np.stack([vectors[i] for i in sorted(vectors)]))
        print(tag, json.dumps(report['sources'][tag]), flush=True)
        (out/'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    report['complete'] = True
    (out/'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--ticks', type=int, default=120)
    parser.add_argument('--start',type=int,default=0,help='TRAIN excerpt start tick; local tracker resets here')
    parser.add_argument('--fast', action='store_true')
    parser.add_argument('--fresh', action='store_true', help='uncached RGB reference for correctness')
    parser.add_argument('--abba', action='store_true', help='four alternating CPU reference/candidate runs')
    args = parser.parse_args()
    if not 50 <= args.ticks <= 200:
        parser.error('bounded TRAIN replay: 50..200 ticks')
    if not 0 <= args.start <= 500:
        parser.error('short fixed TRAIN caches only: start0..500')
    if args.fast and args.fresh:
        parser.error('choose fast or fresh reference')
    if args.abba:
        if args.fast or args.fresh:
            parser.error('abba chooses its own reference/candidate order')
        args.out.mkdir(parents=True, exist_ok=False)
        runs = []
        for index, variant in enumerate(('fresh', 'fast', 'fast', 'fresh')):
            folder = args.out / ('%d_%s' % (index, variant))
            run(folder, args.ticks, variant == 'fast', variant == 'fresh',args.start)
            runs.append(json.loads((folder/'report.json').read_text(encoding='utf-8')))
        summary = {'complete': True, 'order': ['fresh', 'fast', 'fast', 'fresh'],
                   'window': [args.start,args.start+args.ticks], 'sources': {}}
        fields = ('feature_sha256','votes_sha256','view_sha256','ids_sha256','events','tracker_stats')
        for tag in TAGS:
            samples = [r['sources'][tag] for r in runs]
            equal = {key: all(s[key] == samples[0][key] for s in samples) for key in fields}
            summary['sources'][tag] = {'equal': equal,
                'reference_feature_mean_ms': float(np.mean([samples[i]['feature_service']['mean_ms'] for i in (0,3)])),
                'fast_feature_mean_ms': float(np.mean([samples[i]['feature_service']['mean_ms'] for i in (1,2)])),
                'reference_total_seconds': [samples[i]['total_seconds'] for i in (0,3)],
                'fast_total_seconds': [samples[i]['total_seconds'] for i in (1,2)],
                'reference_wait_p90_ms': [samples[i]['feature_wait']['p90_ms'] for i in (0,3)],
                'fast_wait_p90_ms': [samples[i]['feature_wait']['p90_ms'] for i in (1,2)]}
            if not all(equal.values()):
                summary['complete'] = False
        (args.out/'report.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
        if not summary['complete']:
            raise RuntimeError('before/after equivalence failed; inspect saved features')
    else:
        run(args.out, args.ticks, args.fast, args.fresh,args.start)
