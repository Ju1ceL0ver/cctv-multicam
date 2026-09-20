"""Day-qualified clip keys; existing reviewed clips remain at their original paths.

Also the book of windows that keep failing. One window that cannot be labelled (an empty
stretch, a cut-off file) used to hold the whole night queue: the keeper always went back
to the oldest day that was not fully covered, so every later day waited behind it."""
from datetime import datetime
from pathlib import Path
from storage import atomic_json, file_lock, read_json

MAX_FAILURES = 2          # a window that fails this many times steps aside ...
RETRY_AFTER_HOURS = 20    # ... and is tried once more the next night, not on every pass


def clip_key(day, start, clips):
    new = 'c%s_%s' % (day, start.strftime('%H%M%S'))
    old = 'c' + start.strftime('%H%M%S')
    for key in (new, old):
        folder = Path(clips) / key
        for p in folder.glob('meta_*.json'):
            meta = read_json(p, {})
            if meta.get('day') == day and meta.get('start', '').startswith(start.isoformat(timespec='seconds')):
                return key
    return new


def clip_ready(folder, tag='yolo26x-seg'):
    """A people file alone is not proof that crops and review groups were exported."""
    folder = Path(folder)
    return all((folder / (name + '_' + tag + '.json')).exists()
               for name in ('people', 'pieces', 'groups'))


def _book(root):
    return Path(root) / 'data' / 'window_failures.json'


def load_failures(root):
    return read_json(_book(root), {})


def record_failure(root, clip, day, stage, error='', now=None):
    """One more failed attempt on a window; returns how many attempts have failed."""
    now = now or datetime.now()
    path = _book(root)
    with file_lock(str(path) + '.lock'):
        book = read_json(path, {})
        attempts = int(book.get(clip, {}).get('attempts', 0)) + 1
        book[clip] = {'day': day, 'stage': stage, 'attempts': attempts,
                      'last': now.isoformat(timespec='seconds'), 'error': (error or '')[-600:]}
        atomic_json(path, book)
    return attempts


def clear_failure(root, clip):
    path = _book(root)
    if not path.exists():
        return
    with file_lock(str(path) + '.lock'):
        book = read_json(path, {})
        if book.pop(clip, None) is not None:
            atomic_json(path, book)


def set_aside(entry, now=None):
    """True while a window that keeps failing must not hold the queue or eat the night."""
    if not entry or int(entry.get('attempts', 0)) < MAX_FAILURES:
        return False
    now = now or datetime.now()
    return (now - datetime.fromisoformat(entry['last'])).total_seconds() < RETRY_AFTER_HOURS * 3600
