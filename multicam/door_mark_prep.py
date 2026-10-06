"""Light copies of every door stretch video for /doormark, made ahead (CPU, 06.10.2026)."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import door_mark as DM  # noqa: E402

for day in sys.argv[1:] or ['20260919', '20260917', '20260918']:
    for s in DM.stretches(day):
        DM.video(s['tag'])
    print(day, 'ready', flush=True)
