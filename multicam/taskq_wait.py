"""A queue task that waits: until every given status file says finished, or a log line appears.
usage: taskq_wait.py status:PATH ... | log:PATH:TEXT ..."""
import json
import sys
import time


def ok(spec):
    kind, rest = spec.split(':', 1)
    try:
        if kind == 'status':
            return 'finished' in json.load(open(rest))
        path, text = rest.rsplit(':', 1)
        return text in open(path, encoding='utf-8', errors='ignore').read()
    except Exception:
        return False


while not all(ok(s) for s in sys.argv[1:]):
    time.sleep(60)
