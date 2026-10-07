"""07.10: two frames of door_20260919_27663 (the owner's entry of T14 at 36.0 s and the model's at 57.6 s), SAM 3.1's
outlines, T14 (pieces 14, 17, 2, 12) in yellow with the binary model's p_inside, the others by their marked side."""
import json, sys
from pathlib import Path
import cv2, numpy as np
R = Path(__file__).resolve().parent
sys.path.insert(0, str(R)); sys.path.insert(0, str(R / 'inout_lab'))
import door_side as S
from binary_door import BinaryDoorClassifier
import door_mark as DM
clf = BinaryDoorClassifier()
tag, day = 'door_20260919_27663', '20260919'
st = S.migrate(tag, S.load(day)[tag])
ol = S.outlines(tag)
T14 = {'14', '17', '2', '12'}
cap = cv2.VideoCapture(str(R / 'data' / 'sam31_door' / tag / 'cam1' / 'video.mp4'))
outs = []
for t, title in ((36.0, 'ТЫ: T14 вошёл (36.0 с)'), (57.6, 'МОДЕЛЬ: T14 вошёл (57.6 с)')):
    k = int(round(t * 12.5))
    cap.set(cv2.CAP_PROP_POS_FRAMES, k)
    ok, f = cap.read()
    img = cv2.resize(f, (1280, 720))
    rgb = f[:, :, ::-1].copy()
    for piece, person, poly in ol.get(str(k), []):
        p = S.root_of(st.get('merge', {}), str(piece))
        if p in st.get('noperson', []):
            continue
        pts = np.array(poly, np.int32)
        is14 = str(piece) in T14 or p in T14
        xs = [x for x in S.events(st['sides'], st.get('merge'), st.get('noperson', []))]
        col = (0, 255, 255) if is14 else (200, 200, 200)
        txt = 'T%s' % p
        if is14:
            m = np.zeros(f.shape[:2], np.uint8)
            cv2.fillPoly(m, [(pts * [2176 / 1280, 1224 / 720]).astype(np.int32)], 1)
            pr = clf.predict_rgb(rgb, m)['p_inside']
            txt = 'T14  p(внутри)=%.2f' % pr
        cv2.polylines(img, [pts], True, col, 4)
        top = pts[pts[:, 1].argmin()]
        cv2.putText(img, txt, (int(top[0]) - 60, max(30, int(top[1]) - 12)), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 7)
        cv2.putText(img, txt, (int(top[0]) - 60, max(30, int(top[1]) - 12)), cv2.FONT_HERSHEY_SIMPLEX, 1.0, col, 2)
    cv2.rectangle(img, (0, 0), (1280, 44), (0, 0, 0), -1)
    cv2.putText(img, title, (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    outs.append(img)
cv2.imwrite(str(R / 'data' / 'logs' / 'shot_t14.jpg'), np.vstack(outs), [cv2.IMWRITE_JPEG_QUALITY, 88])
print('ok')
