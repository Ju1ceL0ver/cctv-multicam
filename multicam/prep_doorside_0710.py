import json, sys
from pathlib import Path
R = Path(__file__).resolve().parent
sys.path.insert(0, str(R))
import door_mark as DM, door_side as S
day = sys.argv[1] if len(sys.argv) > 1 else '20260919'
st = S.load(day)
xs = [x for x in DM.stretches(day) if not st.get(x['tag'], {}).get('done')]
xs.sort(key=lambda x: -len([a for a in x['answers'] if a['kind'] in ('in', 'out')]))
pick = xs[:12]
for x in pick:
    S.outlines(x['tag']); DM.video(x['tag'])
    print(x['tag'], len([a for a in x['answers'] if a['kind'] in ('in', 'out')]), round(x['seconds']), flush=True)
