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
        pieces = json.load(open(os.path.join(d, 'pieces_%s.json' % TAG)))
        dets, feats, meta, embs = load(clip, TAG)
        piece_of = {'cam1': {}, 'cam2': {}}
        for p in pieces:
            for i in p['dets']:
                piece_of[p['cam']][int(i)] = p['piece']
        per_person = defaultdict(lambda: {'cam1': [], 'cam2': []})
        for cam in ('cam1', 'cam2'):
            for k, who in gt[cam].items():
                i = int(k)
                if i < len(dets[cam]):
                    per_person[who][cam].append((float(dets[cam][i, 0]), i))
        rows = []
        for who, by_cam in sorted(per_person.items()):
            breaks_seen, breaks_gone, gaps, total, seen_pieces = 0, 0, [], 0, set()
            for cam, items in by_cam.items():
                items.sort()
                total += len(items)
                prev_piece, prev_t = None, None
                for t, i in items:
                    pc = piece_of[cam].get(i)
                    seen_pieces.add((cam, pc))
                    if prev_piece is not None and pc != prev_piece:
                        gap = t - prev_t
                        gaps.append(gap)
                        if gap <= 4.0 / FPS:       # the camera never lost them, yet the piece ended
                            breaks_seen += 1
                        else:
                            breaks_gone += 1
                    prev_piece, prev_t = pc, t
            rows.append((total, who, len(seen_pieces - {(c, None) for c in ('cam1', 'cam2')}),
                         breaks_seen, breaks_gone, float(np.median(gaps)) if gaps else 0.0))
        rows.sort(reverse=True)
        print('== %s' % clip)
        for dets_n, who, n_pieces, seen, gone, med in rows[:8]:
            print('   %-4s %6d detections in %3d pieces | breaks: %3d while still visible, %3d after losing them'
                  ' | typical gap %.1f s' % (who, dets_n, n_pieces, seen, gone, med))
        tot_seen = sum(r[3] for r in rows); tot_gone = sum(r[4] for r in rows)
        print('   all people: %d breaks while visible, %d after losing sight -- %.0f%% are ours to fix'
              % (tot_seen, tot_gone, 100 * tot_seen / max(1, tot_seen + tot_gone)), flush=True)


if __name__ == '__main__':
    main()
