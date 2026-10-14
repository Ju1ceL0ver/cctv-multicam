"""Contact sheet of a day's live door events with the owner's /visitpairs answer on each (for review by eye).
usage: visits_sheet.py DAY -> data/visits/<day>/sheet_<n>.jpg (36 per sheet) + sheet.json"""
import json, os, sys
import cv2, numpy as np
ROOT = os.path.dirname(os.path.abspath(__file__))
day = sys.argv[1]
ans = json.load(open(os.path.join(ROOT, 'data', 'visits', day, 'people_answers.json'), encoding='utf-8'))
staff = set(ans.pop('_staff', []))
ev = [json.loads(l) for l in open(os.path.join(ROOT, 'data', 'live', 'events_cam1.jsonl'), encoding='utf-8')]
ev = sorted([e for e in ev if e.get('engine') == 'doorseg_c+zones' and e['clock'].replace('-', '').startswith(day)], key=lambda e: e['t'])
tiles, meta = [], []
for i, e in enumerate(ev):
    a = ans.get('%.1f' % e['t'])
    img = cv2.imread(e.get('photo') or '') if e.get('photo') else None
    if img is None:
        img = np.zeros((720, 1280, 3), np.uint8)
    img = cv2.resize(img, (426, 240))
    lab = 'X' if a == 'x' else ('?' if a is None else 'P%s%s' % (a, 's' if a in staff else ''))
    col = (0, 0, 255) if a == 'x' else (200, 200, 200) if a is None else (0, 220, 0)
    cv2.rectangle(img, (0, 0), (426, 24), (0, 0, 0), -1)
    cv2.putText(img, '%d %s %s %s' % (i, e['kind'].upper(), e['clock'][11:19], lab), (4, 18), cv2.FONT_HERSHEY_SIMPLEX, .6, col, 2)
    tiles.append(img); meta.append({'i': i, 'kind': e['kind'], 'clock': e['clock'], 'label': lab, 'photo': e.get('photo')})
out = os.path.join(ROOT, 'data', 'visits', day)
for s in range(0, len(tiles), 36):
    T = tiles[s:s + 36] + [np.zeros((240, 426, 3), np.uint8)] * (-len(tiles[s:s + 36]) % 6)
    rows = [np.hstack(T[r:r + 6]) for r in range(0, len(T), 6)]
    cv2.imwrite(os.path.join(out, 'sheet_%d.jpg' % (s // 36)), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 80])
json.dump(meta, open(os.path.join(out, 'sheet.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=0)
print(len(ev), 'events', sum(m['label'] == 'X' for m in meta), 'X', sum(m['label'] == '?' for m in meta), 'unlabelled')
