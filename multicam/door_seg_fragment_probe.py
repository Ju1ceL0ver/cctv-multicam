"""Bounded CPU diagnosis of unchanged association on completed owner TRAIN.

Uses trusted local prediction caches and the existing greedy mask match (.5).
Teacher masks with owner identities are diagnostic references, not independent
pixel ground truth. Does not import torch, decode video, fit thresholds or train.
"""
import argparse
from collections import Counter
import hashlib
import json
import pickle
import time
from pathlib import Path

import numpy as np

from door_seg_track import Tracker, centre

ROOT = Path(__file__).resolve().parent
TAGS = ('door_20260918_01405', 'door_20260918_15375')


def unpack(row):
    return (np.unpackbits(row[0], count=184*320).reshape(184, 320).astype(bool), *row[1:])


def matches(pred, truth):
    # Preserve door_seg._match's greedy ordering, including ties; avoid its GPU imports.
    pairs = []
    for i, p in enumerate(pred):
        for j, t in enumerate(truth):
            inter = np.logical_and(p[0], t[0]).sum()
            if inter:
                pairs.append((float(inter / np.logical_or(p[0], t[0]).sum()), i, j))
    out, used_pred, used_truth = [], set(), set()
    for iou, i, j in sorted(pairs, reverse=True):
        if iou < .5:
            break
        if i not in used_pred and j not in used_truth:
            used_pred.add(i)
            used_truth.add(j)
            out.append((i, j, iou))
    return out, pairs


class MeasuredTracker(Tracker):
    def cost_matrix(self, old, predictions, tick):
        self.old = list(old)
        before = time.perf_counter()
        self.costs = super().cost_matrix(old, predictions, tick)
        self.cost_seconds += time.perf_counter() - before
        self.pairs += len(old) * len(predictions)
        return self.costs

    def __init__(self):
        super().__init__()
        self.cost_seconds, self.pairs = 0., 0


def mask_iou(a, b):
    inter = int(np.logical_and(a, b).sum())
    return inter / max(1, int(np.logical_or(a, b).sum()))


def rejection(tracker, previous_pid, column, pred, tick, linked, labels, track_labels):
    old = tracker.old
    row = next((i for i, t in enumerate(old) if t.pid == previous_pid), None)
    if row is None:
        return {'reason': 'previous_id_expired'}
    tr, c = old[row], tracker.config
    gap = tick - tr.tick
    before, now = centre(tr.pred[1]), centre(pred[1])
    height = max(16., min(tr.pred[1][3]-tr.pred[1][1], pred[1][3]-pred[1][1]))
    displacement = float(np.linalg.norm(now-before))
    cost = float(tracker.costs[row, column])
    detail = {'gap': gap, 'cost': cost if np.isfinite(cost) else None,
              'displacement_px': displacement, 'previous_box': np.asarray(tr.pred[1]).tolist(),
              'current_box': np.asarray(pred[1]).tolist()}
    if displacement > c.max_step * gap:
        return dict(detail, reason='physical_step_gate')
    if gap == 1:
        forward = float(np.linalg.norm(now-(before+np.asarray(tr.pred[4]))))
        backward = float(np.linalg.norm(before-(now+np.asarray(pred[3]))))
        gate = max(8., c.centre_gate * height)
        detail.update(forward_error_px=forward, backward_error_px=backward, gate_px=gate,
                      forward_offset=np.asarray(tr.pred[4]).tolist(), backward_offset=np.asarray(pred[3]).tolist())
        if max(forward, backward) > gate:
            return dict(detail, reason='both_offset_gates' if min(forward, backward) > gate else
                        'forward_offset_gate' if forward > gate else 'backward_offset_gate')
    else:
        gate = min(c.max_step*gap, (c.centre_gate+c.gap_growth*gap)*height)
        error = float(np.linalg.norm(now-(before+tr.velocity*gap)))
        detail.update(gate_px=gate, velocity_error_px=error)
        if error > gate:
            return dict(detail, reason='gap_velocity_gate')
    if cost > c.max_cost:
        return dict(detail, reason='max_cost_gate')
    adjacent = gap == 1
    rows = [i for i, t in enumerate(old) if (tick-t.tick == 1) == adjacent]
    # Reconstruct precisely the columns available to the second (gap) stage.
    adjacent_pids = {t.pid for t in old if tick-t.tick == 1}
    cols = list(range(len(linked))) if adjacent else [j for j, (pid, _) in enumerate(linked) if pid not in adjacent_pids]
    if column not in cols:
        return dict(detail, reason='claimed_by_adjacent_track')
    sub = tracker.costs[np.ix_(rows, cols)]
    i, j = rows.index(row), cols.index(column)
    rival = min(np.min(np.delete(sub[i], j), initial=np.inf),
                np.min(np.delete(sub[:, j], i), initial=np.inf))
    detail['rival_cost'] = float(rival) if np.isfinite(rival) else None
    if rival <= c.max_cost and rival-cost < c.ambiguity:
        others = [(float(sub[ii, j]), 'track', ii) for ii in range(len(rows)) if ii != i]
        others += [(float(sub[i, jj]), 'prediction', jj) for jj in range(len(cols)) if jj != j]
        _, kind, at = min(others)
        if kind == 'track':
            other = old[rows[at]]
            known = track_labels.get(other.pid)
            detail.update(rival_kind=kind, rival_pid=other.pid, rival_tick=other.tick,
                          rival_owner=str(known[1]) if known and known[0] == other.tick else None,
                          rival_mask_iou_to_previous=mask_iou(other.pred[0], tr.pred[0]),
                          rival_box=np.asarray(other.pred[1]).tolist())
        else:
            other = linked[cols[at]][1]
            detail.update(rival_kind=kind, rival_prediction_row=cols[at],
                          rival_owner=str(labels[cols[at]]) if cols[at] in labels else None,
                          rival_mask_iou_to_current=mask_iou(other[0], pred[0]),
                          rival_box=np.asarray(other[1]).tolist())
        return dict(detail, reason='ambiguity_veto')
    return dict(detail, reason='assignment_competition')


def diagnose(root, tag, start, ticks, exact_dedup=False):
    path = root/'data/quality_backups/doorseg_quick'/(tag+'.pkl')
    raw = path.read_bytes()
    data = pickle.loads(raw)
    owner = json.loads((root/'data/door_side/20260918.json').read_text(encoding='utf-8'))[tag]
    if data['tag'] != tag or not all(s.get('done') and s.get('role') == 'train' for s in (data['state'], owner)):
        raise ValueError('current and cached completed owner TRAIN required: '+tag)
    records = data['records'][start:start+ticks]
    if len(records) != ticks or any(tick != start+j for j, (tick, _, _) in enumerate(records)):
        raise ValueError('contiguous bounded TRAIN cache required')
    tracker = MeasuredTracker()
    counts, reasons, missed = Counter(), Counter(), Counter()
    last_seen, pid_owner, previous_truth, track_labels = {}, {}, set(), {}
    failures, misses, wrong, ids = [], [], [], []
    tracker_seconds = 0.
    dedup_seconds, removed = 0., 0
    started = time.perf_counter()
    for tick, packed, gt in records:
        if time.perf_counter()-started > 60:
            raise TimeoutError('bounded CPU diagnostic expired')
        all_pred, truth = [unpack(z) for z in packed], [unpack(z) for z in gt]
        pred = [z for z in all_pred if z[2] >= .4]
        if exact_dedup:
            from door_seg_duplicates import exact_masks
            before = time.perf_counter()
            original = pred
            pred, retained = exact_masks(original)
            dedup_seconds += time.perf_counter()-before
            removed += len(original)-len(pred)
            union = lambda rows: np.any(np.stack([z[0] for z in rows]), axis=0) if rows else np.zeros((184,320),bool)
            if not np.array_equal(union(original), union(pred)):
                raise AssertionError('foreground union changed')
            if any(p is not original[i] for p, i in zip(pred, retained)):
                raise AssertionError('candidate modified a retained prediction')
        before = time.perf_counter()
        linked = tracker.update(tick, pred)
        tracker_seconds += time.perf_counter()-before
        ids.extend([tick, j, pid] for j, (pid, _) in enumerate(linked))
        mm, pairs = matches(pred, truth)
        all_mm, _ = matches(all_pred, truth)
        labels = {i: truth[j][2] for i, j, _ in mm}
        now_truth = {z[2] for z in truth}
        matched_truth = {j for _, j, _ in mm}
        counts.update(truth=len(truth), predictions=len(pred), matched=len(mm))
        for j, g in enumerate(truth):
            if j in matched_truth:
                continue
            candidates = sorted((iou, i) for iou, i, jj in pairs if jj == j)
            low = any(all_pred[i][2] < .4 for i, jj, _ in all_mm if jj == j)
            reason = ('confidence_filtered' if low else 'no_overlapping_mask' if not candidates else
                      'mask_below_match' if candidates[-1][0] < .5 else 'greedy_match_competition')
            missed[reason] += 1
            if len(misses) < 80:
                misses.append({'tick': tick, 'owner': str(g[2]), 'reason': reason,
                               'best_iou_above_confidence': candidates[-1][0] if candidates else 0.,
                               'teacher_box': np.asarray(g[1]).tolist(), 'teacher_visible_previous': g[2] in previous_truth})
        for i, j, iou in mm:
            who, pid = truth[j][2], linked[i][0]
            if pid in pid_owner and pid_owner[pid] != who:
                counts['observations_on_id_first_matched_to_other_owner'] += 1
                if len(wrong) < 80:
                    wrong.append({'tick': tick, 'owner': str(who), 'pid': pid,
                                  'first_owner': str(pid_owner[pid]), 'teacher_mask_iou': iou})
            pid_owner.setdefault(pid, who)
            if who in last_seen:
                last_tick, old_pid = last_seen[who]
                counts['adjacent_matched_pairs' if tick-last_tick == 1 else 'reappeared_matched_pairs'] += 1
                if old_pid != pid:
                    counts['adjacent_breaks' if tick-last_tick == 1 else 'reappeared_breaks'] += 1
                    detail = rejection(tracker, old_pid, i, pred[i], tick, linked, labels, track_labels)
                    reasons[detail['reason']] += 1
                    if len(failures) < 80:
                        failures.append(dict(detail, tick=tick, owner=str(who), previous_match_tick=last_tick,
                                             previous_pid=old_pid, assigned_pid=pid, teacher_mask_iou=iou))
            last_seen[who] = (tick, pid)
        for i, who in labels.items():
            track_labels[linked[i][0]] = (tick, who)
        previous_truth = now_truth
    if hashlib.sha256(path.read_bytes()).digest() != hashlib.sha256(raw).digest():
        raise ValueError('cache changed during diagnostic')
    return {'ticks': ticks, 'start_tick': start, 'counts': dict(counts), 'break_reasons': dict(reasons),
            'miss_reasons': dict(missed), 'failure_examples': failures, 'miss_examples': misses, 'wrong_examples': wrong,
            'tracker_stats': tracker.stats, 'tracker_seconds': tracker_seconds,
            'cost_matrix_seconds': tracker.cost_seconds, 'cost_pairs': tracker.pairs,
            'exact_duplicates_removed': removed, 'dedup_seconds': dedup_seconds,
            'foreground_union_equal': True if exact_dedup else None,
            'total_seconds': time.perf_counter()-started, 'cache_sha256': hashlib.sha256(raw).hexdigest(),
            'ids_sha256': hashlib.sha256(json.dumps(ids).encode()).hexdigest()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--ticks', type=int, default=200)
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--exact-dedup', action='store_true', help='separate exact-mask candidate, no thresholds changed')
    args = parser.parse_args()
    if not 50 <= args.ticks <= 200 or not 0 <= args.start <= 500:
        parser.error('bounded TRAIN only:50..200 ticks, start0..500')
    args.out.mkdir(parents=True, exist_ok=False)
    report = {'complete': False, 'exact_dedup': args.exact_dedup, 'sources': {}, 'limitations':
              'Cached B baseline; local tracker reset; teacher masks/owner IDs, not independent pixels or events. No TEST, fitting, GPU or production changes.'}
    for tag in TAGS:
        report['sources'][tag] = diagnose(args.root, tag, args.start, args.ticks, args.exact_dedup)
        (args.out/'report.json').write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
        print(tag, json.dumps({k: v for k, v in report['sources'][tag].items() if not k.endswith('_examples')}), flush=True)
    report['complete'] = True
    (args.out/'report.json').write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')


if __name__ == '__main__':
    main()
