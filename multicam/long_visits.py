"""The queue of the visits that actually matter: the people who stayed.

Measured on 17.09: 69% of everything the cameras see lasts under ten seconds. Those are
the passers-by in the mall gallery, and one fragment each is the right answer for them --
the owner's own labelling agrees with the machine there. The people the shop cares about
are the other end of that distribution, and they are exactly the ones the machine breaks
up: of the sixteen reviewed people present 60 s or longer, it made a median of 2 groups,
p90 of 8 and a worst case of 38, all inside a single ten-minute window.

Reviewing clips in order therefore spends most of a session on the fragments that were
never wrong, and produces no ground truth for a whole visit at all. This queue asks about
the long visits instead, and records the two judgements the acceptance criteria keep
separate: whether anyone else is mixed into this visit, and whether the visit is whole.
Neither is inferred from the other, and neither is inferred from the machine agreeing
with itself.
"""
from collections import defaultdict
from pathlib import Path

from day_visits import build_day
from storage import atomic_json, file_lock, read_json

ROOT = Path(__file__).resolve().parent
MINIMUM_SECONDS = 60.0
PURITY = ('clean', 'mixed', 'unsure')          # is anyone else mixed into this visit
WHOLENESS = ('whole', 'partial', 'unsure')     # is anything of this person missing
KEEP_CONTINUATIONS = 3                         # a finite question per visit, not a gallery


def _review_path(day, root):
    return Path(root) / 'data/day_review' / ('%s.json' % day)


def _evidence(keys, by):
    """A judgement belongs to these fragments in this state. Editing any of them retires it:
    an answer about a visit that has since changed is not an answer about this one."""
    return [[key, by[key]['evidence']] for key in sorted(keys)]


def queue(day, minimum=MINIMUM_SECONDS, root=ROOT):
    day_data = build_day(day, root)
    by = {n['key']: n for n in day_data['nodes']}
    stored = {tuple(sorted(r['nodes'])): r
              for r in read_json(_review_path(day, root), {}).get('visits', [])}
    touching = defaultdict(list)
    for proposal in day_data['proposals']:
        touching[proposal['a']].append(proposal)
        touching[proposal['b']].append(proposal)
    rows = []
    for visit in day_data['visits']:
        seconds = visit['last'] - visit['first']
        if seconds < minimum:
            continue
        keys = sorted(visit['nodes'])
        evidence = _evidence(keys, by)
        record = stored.get(tuple(keys))
        fresh = bool(record) and record.get('evidence') == evidence
        inside = set(keys)
        continuations = []
        for key in keys:
            for proposal in touching.get(key, []):
                other = proposal['b'] if proposal['a'] == key else proposal['a']
                if other in inside or other not in by:
                    continue
                continuations.append({'node': other, 'from': key, 'distance': proposal['distance'],
                                      'gap_s': proposal['gap_s'], 'evidence_a': proposal['evidence_a'],
                                      'evidence_b': proposal['evidence_b'],
                                      'a': proposal['a'], 'b': proposal['b']})
        continuations.sort(key=lambda r: (r['distance'], r['gap_s'], r['node']))
        fragments = sorted((by[k] for k in keys), key=lambda n: (n['first'], n['key']))
        rows.append({'id': visit['id'], 'nodes': keys, 'first': visit['first'], 'last': visit['last'],
                     'seconds': round(seconds, 1), 'evidence': evidence,
                     'fragments': [dict(f, seconds=round(f['last'] - f['first'], 1)) for f in fragments],
                     'cams': sorted({c for f in fragments for c in f.get('cams', [])}),
                     'reviewed_fragments': sum(1 for f in fragments if f['reviewed']),
                     'continuations': continuations[:KEEP_CONTINUATIONS],
                     'judgement': record if fresh else None,
                     'retired': bool(record) and not fresh})
    # Unanswered first, longest first: the longest visits carry the most evidence per answer.
    rows.sort(key=lambda r: (r['judgement'] is not None, -r['seconds'], r['id']))
    answered = [r['judgement'] for r in rows if r['judgement']]
    counts = {
        'long_visits': len(rows),
        'answered': len(answered),
        'retired': sum(1 for r in rows if r['retired']),
        # The project's acceptance number, finally countable -- over answered visits only.
        'clean_and_whole': sum(1 for j in answered if j['purity'] == 'clean' and j['wholeness'] == 'whole'),
        'mixed': sum(1 for j in answered if j['purity'] == 'mixed'),
        'partial': sum(1 for j in answered if j['wholeness'] == 'partial'),
        'fragments': sum(len(r['nodes']) for r in rows),
    }
    return {'day': day, 'revision': day_data['revision'], 'minimum': minimum,
            'can_undo': day_data['can_undo'], 'visits': rows, 'counts': counts}


def judge(day, body, root=ROOT):
    """One answer about one visit, against the exact fragments the person was shown."""
    root = Path(root)
    path = _review_path(day, root)
    minimum = float(body.get('minimum', MINIMUM_SECONDS))
    with file_lock(path.with_suffix('.lock')):
        current = read_json(path, {'revision': 0, 'links': []})
        if body.get('revision') != current['revision']:
            raise ValueError('Ответы изменились, обновите список')
        if body.get('purity') not in PURITY or body.get('wholeness') not in WHOLENESS:
            raise ValueError('Неверный ответ о визите')
        fresh = build_day(day, root)
        by = {n['key']: n for n in fresh['nodes']}
        keys = sorted(body.get('nodes') or [])
        if not keys or any(k not in by for k in keys):
            raise ValueError('Фрагменты визита изменились; откройте список заново')
        if not any(sorted(v['nodes']) == keys for v in fresh['visits']):
            raise ValueError('Этот визит больше не собран из тех же фрагментов')
        evidence = _evidence(keys, by)
        if body.get('evidence') != evidence:
            raise ValueError('Фрагменты изменились; проверьте визит заново')
        atomic_json(path.parent / 'history' / ('%s_%08d.json' % (day, current['revision'])), current)
        current['visits'] = [r for r in current.get('visits', []) if sorted(r['nodes']) != keys]
        current['visits'].append({'nodes': keys, 'purity': body['purity'],
                                  'wholeness': body['wholeness'], 'evidence': evidence})
        current['undo_revision'] = current['revision']
        current['revision'] += 1
        atomic_json(path, current)
    return queue(day, minimum, root)
