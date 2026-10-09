"""The live door's crossings -> the Refloor CRM "camera cards", as the old counter sent them (09.10.2026).

Every event of data/live/events_cam1.jsonl after `since` becomes one card: tt NSK-01, camera cam-1, event detect (in) /
left (out), role employee (staff) / guest (customer), time of the crossing, probability of the role, photo (the frame,
person boxed) and crop (door_live.snapshot). The transport is the old counter's own sink
(retail_analytics/sinks/crm_sink.py: Bearer RA_CRM_TOKEN from retail_analytics/.env, retries, the outbox audit trail),
called here one card at a time.

The role comes from the role model a few seconds after the event (records/roles_<day>.jsonl); a card waits for it up
to ROLE_WAIT seconds, then goes with the role the event was told with (customer when none).

Reliability: state.json keeps every card's fate by the event's key, written after each card, so a restart (crash,
reboot -- the keeper starts this again) never sends a card twice and never forgets one. A card the CRM did not get
(network down, 5xx) is tried again every RETRY seconds until it goes, oldest first; one the CRM refused (4xx) is not
retried (it would not fix itself) and is logged. Days in the past are not sent: `since` is set when the sender first
starts (or by --since).

usage: door_crm.py [--dry] [--since UNIX_TIME]      (the keeper runs it when keeper_plan.json has "crm": true)
-> data/live/crm/state.json, data/live/crm/crm_outbox/ (what the CRM was given), data/live/crm/door_crm.log"""
import json
import os
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
LIVE = Path(os.environ.get('RA_LIVE_OUT') or ROOT / 'data' / 'live')
RA = Path(os.environ.get('RA_RETAIL_ROOT') or ROOT.parent / 'retail_analytics')
OUT = LIVE / 'crm'
ROLE_WAIT = 120.0
RETRY = 60.0
POLL = 3.0


def log(msg):
    OUT.mkdir(parents=True, exist_ok=True)
    line = '%s %s' % (time.strftime('%m-%d %H:%M:%S'), msg)
    with open(OUT / 'door_crm.log', 'a', encoding='utf-8') as f:
        f.write(line + '\n')
    print(line, flush=True)


def load_state(since=None):
    p = OUT / 'state.json'
    if p.exists():
        st = json.load(open(p, encoding='utf-8'))
        if since is not None:
            st['since'] = since
        return st
    return {'since': time.time() if since is None else since, 'cards': {}}


def save_state(st):
    from storage import atomic_json
    OUT.mkdir(parents=True, exist_ok=True)
    atomic_json(OUT / 'state.json', st)


def read_events():
    p = LIVE / 'events_cam1.jsonl'
    out = []
    if p.exists():
        for line in p.read_text(encoding='utf-8').splitlines():
            try:
                out.append(json.loads(line))
            except ValueError:
                continue
    return out


def role_of(e, updates, waited):
    """(role, p_staff) as last decided, or None while the role model may still answer"""
    r = updates.get(e['_key'])
    if r and r.get('role') in ('staff', 'customer'):
        return r['role'], float(r.get('p_staff') if r.get('p_staff') is not None else (1.0 if r['role'] == 'staff' else 0.0))
    if e.get('role') in ('staff', 'customer'):
        return e['role'], float(e.get('p_staff') if e.get('p_staff') is not None else (1.0 if e['role'] == 'staff' else 0.0))
    if waited < ROLE_WAIT:
        return None
    return 'customer', 0.0


def crossing(e, role, p_staff):
    """the old counter's CrossingEvent, as crm_sink.card_from_event reads it"""
    return SimpleNamespace(event='entry' if e['kind'] == 'in' else 'exit', unmatched=False, role=role,
                           staff_probability=p_staff, moment=datetime.fromtimestamp(float(e['t'])),
                           snapshot_full=e.get('photo') or '', snapshot_crop=e.get('crop') or '',
                           event_id=uuid.uuid5(uuid.NAMESPACE_URL, 'door_live/' + e['_key']).hex)


def sink(dry):
    sys.path.insert(0, str(RA))
    from retail_analytics.config import load_config
    from retail_analytics.sinks.crm_sink import CameraCardsSink
    cwd = os.getcwd()
    os.chdir(RA)                                       # its .env and configs/default.yaml
    try:
        cfg = load_config()
    finally:
        os.chdir(cwd)
    crm = cfg.crm
    crm.enabled = True                                 # the old service ran with --crm
    crm.dry_run = bool(dry)
    if not crm.token and not dry:
        raise SystemExit('no CRM token (%s in %s)' % (crm.token_env, RA / '.env'))
    return CameraCardsSink(crm, OUT), crm


def main(dry=False, since=None, once=False):
    global OUT
    import live_events as LE
    if dry:                                            # a dry run never marks the real cards as done
        OUT = LIVE / 'crm_dry'
    LE.LIVE = LIVE                                     # the roles file next to the events
    s, crm = sink(dry)
    from retail_analytics.sinks.crm_sink import card_from_event
    held = {}                                          # key -> Card being retried (one outbox record per card)
    st = load_state(since)
    save_state(st)
    log('up: %s -> %s, tt %s, camera %s, events after %s%s' % (
        LIVE / 'events_cam1.jsonl', s.url, crm.tt, crm.camera_code,
        time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(st['since'])), ' (DRY RUN: nothing is sent)' if dry else ''))
    first_seen = {}
    while True:
        now = time.time()
        evs = [dict(e, _key=LE.key_of(e)) for e in read_events() if float(e.get('t', 0)) >= st['since']]
        days = {time.strftime('%Y%m%d', time.localtime(float(e['t']))) for e in evs}
        updates = {}
        for d in sorted(days):
            updates.update(LE.role_updates(d))
        for e in sorted(evs, key=lambda x: float(x['t'])):
            c = st['cards'].get(e['_key'])
            if c and c['status'] in ('sent', 'rejected', 'skipped', 'dry_run'):
                continue
            if c and c['status'] == 'retry' and now - c.get('tried', 0) < RETRY:
                continue
            first_seen.setdefault(e['_key'], now)
            r = role_of(e, updates, now - first_seen[e['_key']])
            if r is None:
                continue
            card = held.get(e['_key']) or card_from_event(crossing(e, *r), crm)
            if card is None:
                st['cards'][e['_key']] = {'status': 'skipped', 'at': time.strftime('%H:%M:%S')}
                save_state(st)
                continue
            ok = s._post(card)
            rec = {'status': 'dry_run' if dry else 'sent' if ok else 'retry', 'role': r[0], 'kind': e['kind'],
                   'clock': e.get('clock'), 'tried': now, 'tries': (c or {}).get('tries', 0) + 1,
                   'at': time.strftime('%H:%M:%S'), 'error': '' if ok else s.last_error}
            if not ok and not dry and card.record is not None:
                try:                                   # the sink marks a 4xx 'failed' in its outbox: never retried
                    if json.loads(Path(card.record).read_text(encoding='utf-8')).get('status') == 'failed':
                        rec['status'] = 'rejected'
                except Exception:
                    pass
            if rec['status'] == 'retry':
                held[e['_key']] = card
            else:
                held.pop(e['_key'], None)
            st['cards'][e['_key']] = rec
            save_state(st)
            log('%s %s %s %s%s' % (rec['status'], e['kind'], e.get('clock', '')[11:], r[0],
                                   '' if ok else ' -- %s' % s.last_error))
        if once:
            return st
        time.sleep(POLL)


if __name__ == '__main__':
    a = sys.argv[1:]
    since = float(a[a.index('--since') + 1]) if '--since' in a else None
    main(dry='--dry' in a, since=since, once='--once' in a)
