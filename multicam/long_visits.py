"""The queue of the visits that actually matter: the people who stayed.

Measured on 17.09: 69% of everything the cameras see lasts under ten seconds. Those are
the passers-by in the mall gallery, and one fragment each is the right answer for them --
the owner's own labelling agrees with the machine there. The people the shop cares about
are the other end of that distribution, and they are exactly the ones the machine breaks
up: of the sixteen reviewed people present 60 s or longer, it made a median of 2 groups,
p90 of 8 and a worst case of 38, all inside a single ten-minute window.

Two judgements are recorded, and the acceptance criteria keep them separate: whether
anyone else is mixed into this visit, and whether the visit is whole.

**Wholeness cannot be asked as a question.** Nobody can confirm that nothing is missing by
looking at what is present; to answer it honestly a person would have to search the whole
day for other pieces of this person. So it is not asked -- it is *closed*. For each visit
this module lists every fragment that ends just before it starts or starts just after it
ends, inside one window, and the person answers one local, answerable question about each:
is this the same person? When none is left unanswered, the edge is closed, and the page can
say what was actually checked -- "nobody comparable appeared in the two minutes after
17:26:20" -- instead of asking the person to guess.

What the window does not cover is stated rather than hidden: a person who left and came
back an hour later is outside it, and a candidate the appearance model calls obviously
different is listed separately instead of being asked about, with its count shown.
"""
from collections import defaultdict
from pathlib import Path

from day_visits import NEIGHBOUR_WINDOW, build_day, decide, neighbours
from storage import atomic_json, file_lock, read_json

ROOT = Path(__file__).resolve().parent
MINIMUM_SECONDS = 60.0
WINDOW_SECONDS = NEIGHBOUR_WINDOW   # the neighbour table is built to exactly this reach
ASK_PER_SIDE = 6                    # a finite question per edge, not a gallery
LIST_PER_SIDE = 3                   # of the rest, only the nearest few travel to the page
# Above this appearance distance a candidate is listed rather than asked about.
#
# Measured on the real 17.09 after this queue went up: of the 2487 fragments sitting at the
# edge of a long visit, the median distance is 0.407 -- two different shoppers in this shop
# are nothing like as far apart as two random descriptors, which is what a first guess of
# 0.5 assumed. At 0.5 the owner would be asked 1477 questions and only 6 of 236 visits
# would close on their own; at 0.30 it is 287 questions and 87 visits close at once.
#
# 0.30 sits above the 0.228 that auto-linking was calibrated to on the same day (109 of 126
# true links), deliberately: a false positive here costs one keystroke, while a miss costs a
# visit wrongly called whole, so the gate is looser than the one that links without asking.
# What it sets aside is counted on screen and reachable with one key, never hidden.
CLEARLY_DIFFERENT = 0.30
PURITY = ('clean', 'mixed', 'unsure')          # is anyone else mixed into this visit
WHOLENESS = ('whole', 'partial', 'unsure')     # is anything of this person missing
# Who the visit is. Measured on the first 58 visits of 17.09: 32 were one employee at his
# desk, a "visit" in nearly every ten-minute window. He is not a shopper, so he is answered
# with one key and kept out of the acceptance number instead of being judged 120 times.
KIND = ('customer', 'staff')
ASSISTANTS = ('claude',)                       # who may have put a suggestion on screen


def _suggestions_path(day, root):
    return Path(root) / 'data/day_review_suggestions' / ('%s.json' % day)


def suggestions(day, root=ROOT):
    """Another labeller's answers, shown to the owner as a proposal and never counted.

    They are keyed by exactly what was judged: a visit by its fragment list, a boundary by
    its pair and both fragments' evidence. Once either changes the proposal simply stops
    matching -- it is never carried over to something it was not made about.
    """
    data = read_json(_suggestions_path(day, root), {})
    return data.get('visits', {}), data.get('links', {}), data.get('source')


def _review_path(day, root):
    return Path(root) / 'data/day_review' / ('%s.json' % day)


def _evidence(keys, by):
    """A judgement belongs to these fragments in this state. Editing any of them retires it:
    an answer about a visit that has since changed is not an answer about this one."""
    return [[key, by[key]['evidence']] for key in sorted(keys)]


def _shots(node):
    """Up to three moments of this fragment, in order: it appears, the middle of it, it
    leaves. Each carries the person's box, so the page asks for a crop and not the whole
    2560x1440 frame."""
    out, seen = [], set()
    for name in ('first_shot', 'representative', 'last_shot'):
        shot = node.get(name)
        if not shot or not shot.get('box'):
            continue
        mark = (shot['cam'], shot['frame'])
        if mark in seen:
            continue
        seen.add(mark)
        out.append({'cam': shot['cam'], 'frame': shot['frame'], 'box': shot['box']})
    return out


def _brief_node(node):
    """Only what the page draws. A fragment's own observation list holds one id per
    detection -- thousands for a long visit -- and the queue used to ship all of them."""
    return {'key': node['key'], 'clip': node['clip'], 'label': node['label'],
            'cams': node['cams'], 'first': node['first'], 'last': node['last'],
            'seconds': round(node['last'] - node['first'], 1), 'reviewed': node['reviewed'],
            'shots': _shots(node), 'portrait': node['pieces'][0] if node['pieces'] else None}


def _trim(candidate, shots):
    """The page draws two crops of a candidate it is asking about and one of a candidate it
    merely lists. Measured on a day of this shape, sending all three of everything made the
    queue 2 MB, nearly all of it pictures of people nobody would ever open."""
    return dict(candidate, shots=candidate['shots'][:shots])


def _side(items):
    """Split one edge into what is asked and what is merely listed, and say if it is closed.

    Asked: unanswered, and not called obviously different by appearance -- nearest look
    first, then nearest in time. Closed means nothing of that kind is left, so the person's
    own answers, not the machine's, are what established the edge. On a day shaped like
    17.09 this asks about one candidate per visit, median, and two at worst.
    """
    items.sort(key=lambda c: (c['gap_s'], c['node']))
    near = [c for c in items if c['decision'] is None
            and (c['distance'] is None or c['distance'] <= CLEARLY_DIFFERENT)]
    near.sort(key=lambda c: (-1.0 if c['distance'] is None else c['distance'], c['gap_s'], c['node']))
    ask = near[:ASK_PER_SIDE]
    asked_nodes = {c['node'] for c in ask}
    rest = [c for c in items if c['node'] not in asked_nodes]
    return {'ask': [_trim(c, 2) for c in ask],
            'rest': [_trim(c, 1) for c in rest[:LIST_PER_SIDE]],
            'more': max(0, len(rest) - LIST_PER_SIDE),
            'total': len(items),
            'answered': sum(1 for c in items if c['decision'] is not None),
            'far': sum(1 for c in items if c['decision'] is None and c['distance'] is not None
                       and c['distance'] > CLEARLY_DIFFERENT),
            'over': len(near) - len(ask),
            'closed': len(near) == 0}


def _candidate(node, row, side, gap, visit_of):
    other = visit_of.get(node['key'])
    return {'node': node['key'], 'clip': node['clip'], 'label': node['label'], 'cams': node['cams'],
            'side': side, 'gap_s': round(gap, 1), 'distance': row['distance'],
            'same_clip': row['same_clip'], 'decision': row['decision'],
            'first': node['first'], 'last': node['last'],
            'seconds': round(node['last'] - node['first'], 1), 'reviewed': node['reviewed'],
            'shots': _shots(node), 'portrait': node['pieces'][0] if node['pieces'] else None,
            'a': row['a'], 'b': row['b'], 'evidence_a': row['evidence_a'], 'evidence_b': row['evidence_b'],
            # Saying "same" here would swallow a whole other visit; the page has to say so.
            'joins': None if other is None or len(other['nodes']) < 2 else
                     {'fragments': len(other['nodes']), 'seconds': round(other['last'] - other['first'], 1)}}


def _visit_proposal(proposal, evidence, proposer):
    if not proposal or proposal.get('evidence') != evidence:
        return None
    return {'kind': proposal.get('kind'), 'purity': proposal.get('purity'),
            'foreign': proposal.get('foreign', []), 'by': proposer,
            'confidence': proposal.get('confidence'), 'notes': proposal.get('notes', '')}


def queue(day, minimum=MINIMUM_SECONDS, root=ROOT):
    day_data = build_day(day, root)
    by = {n['key']: n for n in day_data['nodes']}
    visit_of = {key: visit for visit in day_data['visits'] for key in visit['nodes']}
    around = defaultdict(list)
    for row in neighbours(day, root):
        around[row['a']].append(row)
        around[row['b']].append(row)
    stored = {tuple(sorted(r['nodes'])): r
              for r in read_json(_review_path(day, root), {}).get('visits', [])}
    proposed_visits, proposed_links, proposer = suggestions(day, root)
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
        found = {'before': [], 'after': []}
        seen = set()
        for key in keys:
            for row in around[key]:
                other = row['b'] if row['a'] == key else row['a']
                if other in inside or other in seen or other not in by:
                    continue
                node = by[other]
                # Only the two edges are a question here. A fragment overlapping the visit
                # in time is a different question entirely -- one person is not in two
                # places at once, and every false link measured on 17.09 was of that shape.
                if node['last'] < visit['first']:
                    side, gap = 'before', visit['first'] - node['last']
                elif node['first'] > visit['last']:
                    side, gap = 'after', node['first'] - visit['last']
                else:
                    continue
                if gap > WINDOW_SECONDS:
                    continue
                seen.add(other)
                candidate = _candidate(node, row, side, gap, visit_of)
                proposal = proposed_links.get('%s|%s' % (row['a'], row['b']))
                if (proposal and proposal.get('evidence_a') == row['evidence_a']
                        and proposal.get('evidence_b') == row['evidence_b']):
                    candidate['suggested'] = {'decision': proposal['decision'], 'by': proposer,
                                              'confidence': proposal.get('confidence'),
                                              'notes': proposal.get('notes', '')}
                found[side].append(candidate)
        before, after = _side(found['before']), _side(found['after'])
        fragments = sorted((by[k] for k in keys), key=lambda n: (n['first'], n['key']))
        rows.append({'id': visit['id'], 'nodes': keys, 'first': visit['first'], 'last': visit['last'],
                     'seconds': round(seconds, 1), 'evidence': evidence,
                     'fragments': [_brief_node(f) for f in fragments],
                     'cams': sorted({c for f in fragments for c in f.get('cams', [])}),
                     'reviewed_fragments': sum(1 for f in fragments if f['reviewed']),
                     'before': before, 'after': after,
                     'closed': before['closed'] and after['closed'],
                     'judgement': record if fresh else None,
                     'retired': bool(record) and not fresh,
                     'suggestion': _visit_proposal(proposed_visits.get('|'.join(keys)), evidence, proposer)})
    # Unanswered first, longest first: the longest visits carry the most evidence per answer.
    rows.sort(key=lambda r: (r['judgement'] is not None, -r['seconds'], r['id']))
    answered = [r['judgement'] for r in rows if r['judgement']]
    shoppers = [j for j in answered if j.get('kind', 'customer') == 'customer']
    counts = {
        'long_visits': len(rows),
        'answered': len(answered),
        'staff': len(answered) - len(shoppers),
        'customers_answered': len(shoppers),
        'retired': sum(1 for r in rows if r['retired']),
        # The project's acceptance number, finally countable -- over answered shoppers only.
        'clean_and_whole': sum(1 for j in shoppers if j['purity'] == 'clean' and j['wholeness'] == 'whole'),
        'mixed': sum(1 for j in shoppers if j['purity'] == 'mixed'),
        'partial': sum(1 for j in shoppers if j['wholeness'] == 'partial'),
        'suggested': sum(1 for r in rows if r['suggestion'] and not r['judgement']),
        'fragments': sum(len(r['nodes']) for r in rows),
        'closed': sum(1 for r in rows if r['closed']),
        'open_edges': sum(len(r['before']['ask']) + len(r['after']['ask']) + r['before']['over'] + r['after']['over']
                          for r in rows),
    }
    return {'day': day, 'revision': day_data['revision'], 'minimum': minimum,
            'window': WINDOW_SECONDS, 'can_undo': day_data['can_undo'],
            'visits': rows, 'counts': counts}


def judge(day, body, root=ROOT):
    """One answer about one visit, against the exact fragments the person was shown."""
    root = Path(root)
    path = _review_path(day, root)
    minimum = float(body.get('minimum', MINIMUM_SECONDS))
    with file_lock(path.with_suffix('.lock')):
        current = read_json(path, {'revision': 0, 'links': []})
        if body.get('revision') != current['revision']:
            raise ValueError('Ответы изменились, обновите список')
        kind = body.get('kind', 'customer')
        if kind not in KIND:
            raise ValueError('Неверный ответ о визите')
        if kind == 'customer' and (body.get('purity') not in PURITY or body.get('wholeness') not in WHOLENESS):
            raise ValueError('Неверный ответ о визите')
        assisted = body.get('assisted')
        if assisted is not None and assisted not in ASSISTANTS:
            raise ValueError('Неизвестный источник подсказки')
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
        # Which fragment is the stranger is the thing that makes "mixed" actionable later.
        foreign = sorted(set(body.get('foreign') or []))
        if any(k not in keys for k in foreign):
            raise ValueError('Отмечен фрагмент не из этого визита')
        if foreign and body.get('purity') != 'mixed':
            raise ValueError('Чужие фрагменты отмечают только у смешанного визита')
        record = {'nodes': keys, 'kind': kind,
                  'purity': body['purity'] if kind == 'customer' else None,
                  'wholeness': body['wholeness'] if kind == 'customer' else None,
                  'foreign': foreign if kind == 'customer' else [], 'evidence': evidence}
        # Said out loud in the record: an answer given with a proposal on screen is not the
        # same evidence as one given cold, and agreement between the two can be measured.
        if assisted:
            record['assisted'] = assisted
        atomic_json(path.parent / 'history' / ('%s_%08d.json' % (day, current['revision'])), current)
        current['visits'] = [r for r in current.get('visits', []) if sorted(r['nodes']) != keys]
        current['visits'].append(record)
        current['undo_revision'] = current['revision']
        current['revision'] += 1
        atomic_json(path, current)
    return queue(day, minimum, root)


def answer(day, body, root=ROOT):
    """Everything the queue page saves, through one door: a judgement about a visit, a
    same/different answer about one edge candidate, or undo.

    `brief` returns the counters instead of the whole queue. A "different" answer cannot
    change how the day groups -- it only forbids a merge that was not made, between
    fragments that do not even overlap in time -- so the page keeps its copy and marks that
    one candidate answered, instead of downloading every visit again to redraw one button.
    """
    minimum = float(body.get('minimum', MINIMUM_SECONDS))
    if body.get('action') in ('link', 'undo'):
        decide(day, body, root)
        result = queue(day, minimum, root)
    else:
        result = judge(day, body, root)
    if body.get('brief'):
        return {'day': day, 'brief': True, 'revision': result['revision'],
                'can_undo': result['can_undo'], 'counts': result['counts']}
    return result
