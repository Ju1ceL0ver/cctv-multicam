"""13.10.2026: the live door logic (owner's model + shop line, side_final) on every /doorside stretch, on SAM 3.1 teacher
tracks and on micro1s3 where present -> data/door_v2/zone_baseline.json (per tag counts), to compare the zone rules."""
import json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import door_seg_check as K

out = {}
for f in sorted((ROOT / 'data/door_side').glob('2026*.json')):
    if 'history' in f.name:
        continue
    day = f.stem
    for tag, st in json.load(open(f, encoding='utf-8')).items():
        if not isinstance(st, dict) or not st.get('sides'):
            continue
        item = {'tag': tag, 'day': day, 'state': st}
        res = {'role': st.get('role', 'test')}
        for name in ('sam31', 'micro1s3'):
            base = (ROOT / 'data/sam31_door' / tag / 'cam1') if name == 'sam31' else (ROOT / 'data/micro_door' / name / tag / 'cam1')
            if not (base / 'report.json').exists():
                continue
            if name == 'sam31':
                link = ROOT / 'data/micro_door/_sam31view' / tag
                link.mkdir(parents=True, exist_ok=True)
                import shutil
                for fn in ('report.json', 'info.json', 'chunks.npz', 'ticks.json'):
                    if (base / fn).exists() and not (link / 'cam1' / fn).exists():
                        (link / 'cam1').mkdir(exist_ok=True)
                        shutil.copy(base / fn, link / 'cam1' / fn)
                nm = '_sam31view'
            else:
                nm = name
            try:
                r = K.score_source(item, nm)
                res[name] = {'counts': r['counts'], 'f1': r['f1'], 'events': r['events']}
            except Exception as e:
                res[name] = {'error': str(e)[:200]}
        out[tag] = res
        print(tag, {k: (v.get('counts') if isinstance(v, dict) else v) for k, v in res.items()}, flush=True)
json.dump(out, open(ROOT / 'data/door_v2/zone_baseline.json', 'w'), default=float)
print('done')
