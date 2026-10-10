"""Yield to interactive users and busy foreign apps, not an idle open window."""
import os
import time

_previous = {}
_checked = (0., [])


def active_users():
    if os.name != 'nt':
        return []
    import ctypes as C
    from ctypes import wintypes as W
    class Session(C.Structure):
        _fields_ = [('id', W.DWORD), ('station', W.LPWSTR), ('state', C.c_int)]
    api = C.windll.wtsapi32
    sessions = C.POINTER(Session)(); count = W.DWORD()
    if not api.WTSEnumerateSessionsW(None, 0, 1, C.byref(sessions), C.byref(count)):
        raise OSError('cannot enumerate interactive sessions')
    found = []
    try:
        for i in range(count.value):
            row = sessions[i]
            if row.state != 0:
                continue
            buffer = C.c_void_p(); size = W.DWORD()
            if not api.WTSQuerySessionInformationW(None, row.id, 5, C.byref(buffer), C.byref(size)):
                raise OSError('cannot read active session user')
            try:
                if C.wstring_at(buffer): found.append(row.id)
            finally:
                api.WTSFreeMemory(buffer)
    finally:
        api.WTSFreeMemory(sessions)
    return found


def foreign_busy(rows=None):
    """An attached logged-in user always wins. Disconnected apps are monitored."""
    global _checked
    now = time.monotonic()
    if rows is None and now - _checked[0] < 2:
        return list(_checked[1])
    import psutil
    found = ['interactive session %s' % s for s in active_users()]
    for p in rows if rows is not None else psutil.process_iter(['pid', 'name']):
        if not any(x in (p.info['name'] or '').lower() for x in ('photoshop', '3dsmax', '3ds max')):
            continue
        try:
            cpu = sum(p.cpu_times()[:2]); io = p.io_counters()
            current = (now, cpu, io.read_bytes, io.write_bytes)
            old = _previous.get(p.pid)
            _previous[p.pid] = current
            if old is None:
                found.append('sampling foreign app PID %d' % p.pid)
            else:
                elapsed = max(.01, now-old[0])
                if (cpu-old[1])/elapsed > .2 or (io.write_bytes-old[3])/elapsed > 2**20 or (io.read_bytes-old[2])/elapsed > 4*2**20:
                    found.append('active foreign app PID %d' % p.pid)
        except psutil.Error:
            found.append('foreign activity unreadable PID %d' % p.pid)
    _checked = now, found
    return list(found)
