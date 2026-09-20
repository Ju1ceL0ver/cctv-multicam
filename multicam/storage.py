"""Small atomic JSON stores shared by the review site and background jobs."""
import json
import os
import time
import uuid
from contextlib import contextmanager
from pathlib import Path


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    try:
        with tmp.open('w', encoding='utf-8') as f:
            json.dump(value, f, ensure_ascii=False, indent=1, allow_nan=False)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


@contextmanager
def file_lock(path, timeout=15):
    """OS lock released even after a crash; works across Flask workers and jobs."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    f = path.open('a+b')
    f.seek(0, 2)
    if f.tell() == 0:
        f.write(b'0'); f.flush()
    until = time.monotonic() + timeout
    acquired = False
    try:
        while True:
            try:
                f.seek(0)
                if os.name == 'nt':
                    import msvcrt
                    msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
                else:
                    import fcntl
                    fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
                acquired = True
                break
            except (BlockingIOError, OSError):
                if time.monotonic() >= until:
                    raise TimeoutError('Хранилище занято, повторите сохранение')
                time.sleep(.03)
        yield
    finally:
        try:
            f.seek(0)
            if acquired:
                if os.name == 'nt':
                    import msvcrt
                    msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    import fcntl
                    fcntl.flock(f, fcntl.LOCK_UN)
        finally:
            f.close()


def read_json(path, default=None):
    p = Path(path)
    return json.loads(p.read_text(encoding='utf-8')) if p.exists() else default
