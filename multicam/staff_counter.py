"""The live counter's door snapshots of the days the owner has not labelled (08.10.2026) for /staff: every event of
26.09-05.10 with its crop -> the same teacher ReID vectors as data/teacher_emb (TransReID clothes 3840, CSCI shape
1024) -> data/staff/counter_emb.npz; staff.build adds them as people (src 'counter').
usage (venv_rfdetr, the card or CPU): staff_counter.py [FIRST_DAY] [LAST_DAY]"""
import json, re, sys, time
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
EV = ROOT.parent / 'retail_analytics' / 'runs' / 'live' / 'entrance_events.jsonl'


def main(first='20260926', last='20261005'):
    import cv2
    import track_emb as T
    rows = []
    for l in open(EV, encoding='utf-8'):
        if not l.strip():
            continue
        e = json.loads(l)
        d = time.strftime('%Y%m%d', time.localtime(e['unix_ms'] / 1000))
        if first <= d <= last and e.get('snapshot_crop') and Path(e['snapshot_crop']).exists():
            rows.append((d, e))
    print('events', len(rows), flush=True)
    embed = T.teachers()
    ids, cloth, shape, meta = [], [], [], {}
    for a in range(0, len(rows), 64):
        crops, keep = [], []
        for d, e in rows[a:a + 64]:
            c = cv2.imread(e['snapshot_crop'])
            if c is None or c.shape[0] < 16 or c.shape[1] < 8:
                continue
            crops.append(c); keep.append((d, e))
        if not crops:
            continue
        v1, v2 = embed(crops)
        for (d, e), c1, s1 in zip(keep, v1, v2):
            pid = re.sub(r'[^A-Za-z0-9_]', '_', 'cnt_%s_%s' % (d, e['event_id']))[:90]
            full = cv2.imread(e['snapshot_full']) if e.get('snapshot_full') and Path(e['snapshot_full']).exists() else None
            fh, fw = (full.shape[:2] if full is not None else (720, 1280))
            sx, sy = 1280.0 / fw, 720.0 / fh
            box = [int(e['box_x1'] * sx), int(e['box_y1'] * sy), int(e['box_x2'] * sx), int(e['box_y2'] * sy)]
            ids.append(pid); cloth.append(c1); shape.append(s1)
            meta[pid] = {'src': 'counter', 'day': d, 'cam': 'cam1', 'box': box, 'crop': e['snapshot_crop'],
                         'full': e.get('snapshot_full'), 'event': e['event'], 'counter_role': e.get('role'),
                         'full_size': [fw, fh], 'box_raw': [e['box_x1'], e['box_y1'], e['box_x2'], e['box_y2']]}
        print(a + len(keep), 'of', len(rows), flush=True)
    out = ROOT / 'data' / 'staff'
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / 'counter_emb.npz', ids=np.array(ids), cloth=np.array(cloth, np.float32), shape=np.array(shape, np.float32))
    json.dump(meta, open(out / 'counter_people.json', 'w', encoding='utf-8'))
    print('saved', len(ids), flush=True)


if __name__ == '__main__':
    main(*sys.argv[1:])
