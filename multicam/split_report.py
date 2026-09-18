"""Why a single shopper becomes many pieces.

For every person the owner labelled, this walks their detections in time order and
classifies each break: the camera simply stopped seeing them (a rack, a turn, leaving
the view), or it kept seeing them and the tracker started a new piece anyway. Only the
second kind is ours to fix, and until now we did not know the split."""
import sys, os, json, numpy as np
from collections import defaultdict

ROOT = os.path.dirname(os.path.abspath(__file__)); os.chdir(ROOT); sys.path.insert(0, ROOT)
from clipdata import load, FPS

TAG = 'yolo26x-seg'


def main():
    clips = sys.argv[1:] or ['c103700', 'c112000', 'c155236', 'c183400']
    for clip in clips:
        d = os.path.join('data', 'raw_clips', clip)
        gt = json.load(open(os.path.join(d, 'gt_identity_%s.json' % TAG)))
        dets, feats, meta, embs = load(clip, TAG)
        # Build the tracks now instead of reading the stored pieces: those were cut by
        # whatever tracker ran when the clip was first processed, and the point here is
        # to measure the tracker as it stands today.
        from person3d import Camera
        from fusion2 import build
        cams = {c: Camera(c, json.load(open('data/calib_final.json'))) for c in ('cam1', 'cam2')}
        per_cam, _ = build(cams, dets, feats, embs, meta.get('clean'))
        piece_of = {'cam1': {}, 'cam2': {}}
        for n, it in enumerate(per_cam['cam1'] + per_cam['cam2']):
            for i in it['det']:
                piece_of[it['cam']][int(i)] = n
        per_person = defaultdict(lambda: {'cam1': [], 'cam2': []})
        for cam in ('cam1', 'cam2'):
            for k, who in gt[cam].items():
                i = int(k)
                if i < len(dets[cam]):
                    per_person[who][cam].append((float(dets[cam][i, 0]), i))
        rows = []
        for who, by_cam in sorted(per_person.items()):
            breaks_seen, breaks_gone, gaps, total, seen_pieces, orphan = 0, 0, [], 0, set(), 0
            for cam, items in by_cam.items():
                items.sort()
                total += len(items)
                prev_piece, prev_t = None, None
                for t, i in items:
                    pc = piece_of[cam].get(i)
                    if pc is None:
                        orphan += 1        # dropped as too short or unclean: in no piece at all
                        continue
                    seen_pieces.add((cam, pc))
                    if prev_piece is not None and pc != prev_piece:
                        gap = t - prev_t
                        gaps.append(gap)
                        if gap <= 4.0 / FPS:       # the camera never lost them, yet the piece ended
                            breaks_seen += 1
                        else:
                            breaks_gone += 1
                    prev_piece, prev_t = pc, t
            rows.append((total, who, len(seen_pieces), breaks_seen, breaks_gone,
                         float(np.median(gaps)) if gaps else 0.0, orphan))
        rows.sort(reverse=True)
        print('== %s' % clip)
        for dets_n, who, n_pieces, seen, gone, med, orphan in rows[:8]:
            print('   %-4s %6d detections in %3d pieces | breaks: %3d while visible, %3d after losing them'
                  ' | %d detections in no piece at all (%.0f%%)'
                  % (who, dets_n, n_pieces, seen, gone, orphan, 100.0 * orphan / max(1, dets_n)))
        tot_seen = sum(r[3] for r in rows); tot_gone = sum(r[4] for r in rows)
        tot_orphan = sum(r[6] for r in rows); tot_dets = sum(r[0] for r in rows)
        print('   all people: %d breaks while visible, %d after losing sight, '
              '%d of %d detections (%.0f%%) ended up in no piece'
              % (tot_seen, tot_gone, tot_orphan, tot_dets, 100.0 * tot_orphan / max(1, tot_dets)), flush=True)


if __name__ == '__main__':
    main()
