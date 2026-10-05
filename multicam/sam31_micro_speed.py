"""Real SAM 3.1 modules, shrunk, timed (06.10.2026): Meta's own detector, tracker and rules with the student encoder
in place of the ViT-L and fewer layers (the first k of each stack), on the same crowded 17.09 stretch as
sam31_parts.py. Speed only: the shortened heads are not trained yet, so their masks mean nothing.

configs: full heads + student encoder; detector 2+3 layers, tracker 2; the same without the interactive neck
(only clicks use it).

usage (venv_sam3, the card): sam31_micro_speed.py STUDENT_CKPT [WAIT_FOR_STATUS_JSON]  -> data/logs/sam31_micro_speed.json"""
import json
import shutil
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))


def shrink(pred, det_enc, det_dec, trk):
    import torch.nn as nn
    t = pred.model.detector.transformer
    t.encoder.layers = nn.ModuleList(list(t.encoder.layers)[:det_enc]); t.encoder.num_layers = det_enc
    t.decoder.layers = nn.ModuleList(list(t.decoder.layers)[:det_dec]); t.decoder.num_layers = det_dec
    t.decoder.fine_layers = list(t.decoder.fine_layers)[:det_dec]
    e = pred.model.tracker.model.transformer.encoder
    e.layers = nn.ModuleList(list(e.layers)[:trk]); e.num_layers = trk


def main(ckpt, wait=None):
    import cv2
    import torch
    import sam31_distill as SD
    import sam31_segment as S
    if wait:
        while True:
            try:
                if 'finished' in json.load(open(wait)):
                    break
            except Exception:
                pass
            time.sleep(30)
        time.sleep(20)
    n = 48
    base = sorted((ROOT / 'data' / 'sam31_seg').glob('*/cam1'))[0]
    rows = np.load(base / 'chunks.npz')['rows']
    ticks, cnt = np.unique(rows[:, 1].astype(int), return_counts=True)
    k0 = int(ticks[np.argmax(np.convolve(cnt, np.ones(n), 'same'))]) - n // 2
    sub = ROOT / 'data' / 'logs' / 'micro_speed_frames'
    shutil.rmtree(sub, ignore_errors=True); sub.mkdir(parents=True)
    cap = cv2.VideoCapture(str(base / 'video.mp4'))
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, k0))
    for k in range(n):
        ok, f = cap.read()
        if not ok:
            break
        cv2.imwrite(str(sub / ('%05d.jpg' % k)), cv2.resize(f, (1008, 1008), interpolation=cv2.INTER_AREA))
    student = SD.load_student(ckpt)
    rep = {'frames': n, 'window': base.parent.name, 'configs': {}}
    # the shortened detector, untrained, may find nobody, and an empty tracker costs nothing: so the tracker is timed
    # behind the full detector (real people), the detector in its own config
    for name, cfg in (('full heads + student', None), ('det 6+6, tracker 2', (6, 6, 2)),
                      ('det 2+3, tracker 2, no interactive neck', (2, 3, 2, 'noint'))):
        pred = S.build()
        tri = pred.model.detector.backbone.vision_backbone
        tri.trunk = student
        if cfg:
            shrink(pred, *cfg[:3])
            if len(cfg) > 3:
                fwd = tri.forward
                tri.forward = lambda x, **kw: fwd(x, **dict(kw, need_interactive_out=False))
        m = pred.model
        T = defaultdict(list)

        def wrap(obj, attr, key):
            f = getattr(obj, attr)

            def g(*a, **kw):
                torch.cuda.synchronize(); t = time.perf_counter()
                r = f(*a, **kw)
                torch.cuda.synchronize(); T[key].append(time.perf_counter() - t)
                return r
            setattr(obj, attr, g)
        wrap(m, 'run_backbone_and_detection', 'encoder+detector')
        wrap(m, 'run_tracker_propagation', 'tracker')
        wrap(m, 'run_tracker_update_planning_phase', 'rules+memory')
        wrap(m, 'run_tracker_update_execution_phase', 'new objects')
        wrap(student, 'forward', '  of which encoder')
        nobj = []
        with torch.autocast('cuda', dtype=torch.bfloat16):
            sid = pred.handle_request(dict(type='start_session', resource_path=str(sub), offload_video_to_cpu=True))['session_id']
            pred.handle_request(dict(type='add_prompt', session_id=sid, frame_index=0, text='person'))
            t0 = time.perf_counter()
            for resp in pred.handle_stream_request(dict(type='propagate_in_video', session_id=sid, propagation_direction='forward')):
                nobj.append(len((resp.get('outputs') or {}).get('out_obj_ids', [])))
            total = time.perf_counter() - t0
            pred.handle_request(dict(type='close_session', session_id=sid))
        student.forward = type(student).forward.__get__(student)
        rep['configs'][name] = {'ms_per_frame': round(1000 * total / max(1, len(nobj)), 1),
                                'parts_ms': {k: round(1000 * float(np.median(v[2:] or v)), 1) for k, v in T.items()},
                                'people_mean': round(float(np.mean(nobj)), 1) if nobj else 0}
        print(name, json.dumps(rep['configs'][name]), flush=True)
        del pred
        torch.cuda.empty_cache()
    c = rep['configs']
    try:
        rep['estimate_ms'] = round(c['det 2+3, tracker 2, no interactive neck']['parts_ms']['encoder+detector']
                                   + c['det 6+6, tracker 2']['parts_ms']['tracker'] + c['det 6+6, tracker 2']['parts_ms']['rules+memory'], 1)
    except KeyError:
        pass
    print(json.dumps(rep.get('estimate_ms')), flush=True)
    json.dump(rep, open(ROOT / 'data' / 'logs' / 'sam31_micro_speed.json', 'w'), indent=1)


if __name__ == '__main__':
    main(*sys.argv[1:])
