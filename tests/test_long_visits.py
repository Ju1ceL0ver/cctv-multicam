"""The long-visit queue: what it shows, and what it refuses to accept as an answer."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'multicam'))
from day_visits import build_day                     # noqa: E402
from long_visits import answer, judge, queue         # noqa: E402
from storage import atomic_json, read_json           # noqa: E402

DAY = '20260918'


def _clip(root, name, start, times, label, box=(5, 5, 20, 40), emb=(1., 0.), extra=()):
    """One clip. `extra` adds more people to the same ten-minute window -- the place the
    measurements say a long visit actually breaks. A label of None leaves the person as an
    unreviewed machine group (`G0`, `G1`, ...), which is what those breaks look like."""
    people = [(label, list(times), tuple(box), tuple(emb))] + [tuple(p) for p in extra]
    d = root / 'data/raw_clips' / name
    d.mkdir(parents=True)
    rows, embeddings, pieces, groups, labels = [], [], [], [], {}
    for index, (name_, moments, where, feature) in enumerate(people):
        first = len(rows)
        for moment in moments:
            row = np.zeros(10, np.float32)
            row[0] = moment
            row[1:5] = where
            rows.append(row)
            embeddings.append(feature)
        pieces.append({'piece': index, 'cam': 'cam1', 't0': float(moments[0]),
                       't1': float(moments[-1]), 'dets': list(range(first, len(rows)))})
        groups.append({'person': index, 'pieces': [index]})
        if name_ is not None:
            labels[str(index)] = name_
    atomic_json(d / 'meta_yolo26x-seg.json', {'day': DAY,
        'seconds': max(120.0, max(float(m) for _, ms, _, _ in people for m in ms) + 1),
        'start': '2026-09-18T%s' % start, 'width': 100, 'height': 80})
    atomic_json(d / 'pieces_yolo26x-seg.json', pieces)
    atomic_json(d / 'groups_yolo26x-seg.json', groups)
    atomic_json(d / 'gt_manual.json', labels)
    np.savez(d / 'dets_yolo26x-seg.npz', cam1=np.array(rows, np.float32), cam2=np.empty((0, 10)))
    np.savez(d / 'emb_osnet_ain_x1_0_msmt17.npz',
             cam1=np.array(embeddings, np.float64), cam2=np.empty((0, 2)))
    return d


@pytest.fixture
def day(tmp_path):
    """One person of just under two minutes, seen in two overlapping clips (so the two
    fragments join on shared raw frames), and one passer-by of three seconds."""
    root = tmp_path / 'project'
    _clip(root, 'c_early', '10:00:00', [0, 60, 61, 119], 'P1')
    _clip(root, 'c_late', '10:01:00', [0, 1, 59], 'P1')
    _clip(root, 'c_short', '10:30:00', [0, 1.5, 3], 'P7')
    return root


def test_only_the_people_who_stayed_are_asked_about(day):
    result = queue(DAY, root=day)
    assert [v['nodes'] for v in result['visits']] == [['c_early:P1', 'c_late:P1']]
    visit = result['visits'][0]
    assert visit['seconds'] == 119.0 and len(visit['fragments']) == 2
    assert [f['clip'] for f in visit['fragments']] == ['c_early', 'c_late']
    assert result['counts'] == {'long_visits': 1, 'answered': 0, 'retired': 0, 'staff': 0,
                                'customers_answered': 0, 'suggested': 0,
                                'clean_and_whole': 0, 'mixed': 0, 'partial': 0, 'fragments': 2,
                                'closed': 1, 'open_edges': 0}


def test_a_shorter_minimum_brings_the_passer_by_back(day):
    assert len(queue(DAY, minimum=2, root=day)['visits']) == 2
    assert len(queue(DAY, minimum=1000, root=day)['visits']) == 0


def test_an_answer_is_recorded_and_counted_as_the_acceptance_number(day):
    fresh = queue(DAY, root=day)
    visit = fresh['visits'][0]
    after = judge(DAY, {'revision': fresh['revision'], 'nodes': visit['nodes'],
                        'evidence': visit['evidence'], 'purity': 'clean', 'wholeness': 'whole'}, root=day)
    assert after['counts']['answered'] == 1 and after['counts']['clean_and_whole'] == 1
    assert after['visits'][0]['judgement']['purity'] == 'clean'
    assert after['revision'] == fresh['revision'] + 1 and after['can_undo']
    stored = read_json(day / 'data/day_review' / (DAY + '.json'))
    assert stored['visits'][0]['nodes'] == visit['nodes']


def test_a_mixed_visit_is_not_counted_as_correct(day):
    fresh = queue(DAY, root=day)
    visit = fresh['visits'][0]
    after = judge(DAY, {'revision': fresh['revision'], 'nodes': visit['nodes'],
                        'evidence': visit['evidence'], 'purity': 'mixed', 'wholeness': 'partial'}, root=day)
    counts = after['counts']
    assert counts['answered'] == 1 and counts['clean_and_whole'] == 0
    assert counts['mixed'] == 1 and counts['partial'] == 1


def test_answering_again_replaces_the_previous_answer(day):
    fresh = queue(DAY, root=day)
    visit = fresh['visits'][0]
    after = judge(DAY, {'revision': fresh['revision'], 'nodes': visit['nodes'],
                        'evidence': visit['evidence'], 'purity': 'clean', 'wholeness': 'whole'}, root=day)
    again = judge(DAY, {'revision': after['revision'], 'nodes': visit['nodes'],
                        'evidence': visit['evidence'], 'purity': 'unsure', 'wholeness': 'unsure'}, root=day)
    assert again['counts']['answered'] == 1 and again['counts']['clean_and_whole'] == 0
    assert len(read_json(day / 'data/day_review' / (DAY + '.json'))['visits']) == 1


def test_editing_a_fragment_retires_the_answer_instead_of_keeping_it(day):
    fresh = queue(DAY, root=day)
    visit = fresh['visits'][0]
    judge(DAY, {'revision': fresh['revision'], 'nodes': visit['nodes'], 'evidence': visit['evidence'],
                'purity': 'clean', 'wholeness': 'whole'}, root=day)
    atomic_json(day / 'data/raw_clips/c_late/pieces_yolo26x-seg.json',
                [{'piece': 0, 'cam': 'cam1', 't0': 0.0, 't1': 1.0, 'dets': [0, 1]}])
    after = queue(DAY, root=day)
    row = after['visits'][0]
    assert row['judgement'] is None and row['retired']
    assert after['counts']['answered'] == 0 and after['counts']['retired'] == 1


@pytest.mark.parametrize('change,message', [
    ({'revision': 99}, 'обновите список'),
    ({'purity': 'probably'}, 'Неверный ответ'),
    ({'wholeness': 'mostly'}, 'Неверный ответ'),
    ({'nodes': ['c_early:P1']}, 'больше не собран'),
    ({'nodes': ['c_early:P1', 'c_missing:P4']}, 'откройте список заново'),
    ({'nodes': []}, 'откройте список заново'),
    ({'evidence': [['c_early:P1', 'stale'], ['c_late:P1', 'stale']]}, 'проверьте визит заново'),
])
def test_an_answer_about_something_else_is_refused(day, change, message):
    fresh = queue(DAY, root=day)
    visit = fresh['visits'][0]
    body = {'revision': fresh['revision'], 'nodes': visit['nodes'], 'evidence': visit['evidence'],
            'purity': 'clean', 'wholeness': 'whole'}
    body.update(change)
    with pytest.raises(ValueError, match=message):
        judge(DAY, body, root=day)
    assert queue(DAY, root=day)['counts']['answered'] == 0


def test_a_fragment_already_in_the_visit_is_never_offered_at_its_edges(day):
    visit = queue(DAY, root=day)['visits'][0]
    offered = {c['node'] for side in ('before', 'after') for c in visit[side]['ask']}
    assert not (offered & set(visit['nodes']))


def test_unanswered_visits_come_first_and_the_longest_lead(tmp_path):
    root = tmp_path / 'project'
    _clip(root, 'c_one', '10:00:00', [0, 70], 'P1')
    _clip(root, 'c_two', '11:00:00', [0, 100], 'P2')
    fresh = queue(DAY, root=root)
    assert [v['fragments'][0]['clip'] for v in fresh['visits']] == ['c_two', 'c_one']
    longest = fresh['visits'][0]
    after = judge(DAY, {'revision': fresh['revision'], 'nodes': longest['nodes'],
                        'evidence': longest['evidence'], 'purity': 'clean', 'wholeness': 'whole'}, root=root)
    assert [v['fragments'][0]['clip'] for v in after['visits']] == ['c_one', 'c_two']


def test_the_queue_reports_the_same_revision_the_answer_must_quote(day):
    fresh = queue(DAY, root=day)
    assert fresh['revision'] == build_day(DAY, day)['revision']


def test_the_page_and_its_api_are_served(day, monkeypatch):
    import label_pieces as web
    monkeypatch.setattr(web, 'ROOT', str(day))
    monkeypatch.setattr(web, 'CLIPS', str(day / 'data/raw_clips'))
    web.app.config['TESTING'] = True
    client = web.app.test_client()
    client.set_cookie('labeler_key', web.KEY)
    assert client.get('/longvisits').status_code == 200
    assert client.get('/static/longvisits.js').status_code == 200
    assert client.get('/api/day/not-a-day/visits').status_code == 400
    listing = client.get('/api/day/%s/visits' % DAY)
    assert listing.status_code == 200 and len(listing.json['visits']) == 1
    assert client.get('/api/day/%s/visits?seconds=1000' % DAY).json['visits'] == []
    visit = listing.json['visits'][0]
    answer = {'revision': listing.json['revision'], 'nodes': visit['nodes'],
              'evidence': visit['evidence'], 'purity': 'clean', 'wholeness': 'whole'}
    assert client.post('/api/day/%s/visits' % DAY, json={**answer, 'revision': 99}).status_code == 409
    saved = client.post('/api/day/%s/visits' % DAY, json=answer)
    assert saved.status_code == 200 and saved.json['counts']['clean_and_whole'] == 1


@pytest.fixture
def edges(tmp_path):
    """A four-minute visit with exactly the neighbours the machine leaves around it: a
    second machine group thirteen seconds later *inside the same ten-minute window* (the
    measured main cause of a broken long visit), somebody else present at the same time,
    and somebody six minutes later."""
    root = tmp_path / 'project'
    _clip(root, 'c_main', '10:00:00', [0, 120, 240], 'P1',
          extra=[(None, [253, 300], (5, 5, 20, 40), (1., 0.))])
    _clip(root, 'c_over', '10:02:00', [0, 60], 'P2', box=(50, 5, 70, 40))
    _clip(root, 'c_far', '10:10:00', [0, 60], 'P3')
    return root


def _visit(result, key):
    return next(v for v in result['visits'] if key in v['nodes'])


def test_the_end_of_a_visit_is_a_list_of_who_came_next_not_a_guess(edges):
    visit = _visit(queue(DAY, root=edges), 'c_main:P1')
    assert visit['seconds'] == 240.0
    assert [c['node'] for c in visit['after']['ask']] == ['c_main:G1']
    candidate = visit['after']['ask'][0]
    assert candidate['gap_s'] == 13.0 and candidate['side'] == 'after'
    # The break is inside one window, where nothing automatic ever proposes a link.
    assert candidate['same_clip'] and candidate['decision'] is None
    assert visit['before']['closed'] and not visit['after']['closed'] and not visit['closed']


def test_somebody_present_at_the_same_time_is_not_a_boundary_question(edges):
    visit = _visit(queue(DAY, root=edges), 'c_main:P1')
    offered = {c['node'] for side in ('before', 'after') for c in visit[side]['ask'] + visit[side]['rest']}
    assert 'c_over:P2' not in offered        # one person is not in two places at once
    assert 'c_far:P3' not in offered         # six minutes away is outside the window


def test_saying_different_closes_the_edge_and_the_visit(edges):
    fresh = queue(DAY, root=edges)
    candidate = _visit(fresh, 'c_main:P1')['after']['ask'][0]
    after = answer(DAY, {'action': 'link', 'decision': 'different', 'revision': fresh['revision'],
                         'a': candidate['a'], 'b': candidate['b'],
                         'evidence_a': candidate['evidence_a'], 'evidence_b': candidate['evidence_b']},
                   root=edges)
    visit = _visit(after, 'c_main:P1')
    assert visit['after']['closed'] and visit['closed']
    assert visit['after']['answered'] == 1 and visit['after']['ask'] == []
    assert visit['after']['rest'][0]['decision'] == 'different'


def test_saying_same_grows_the_visit_and_moves_its_edge(edges):
    fresh = queue(DAY, root=edges)
    candidate = _visit(fresh, 'c_main:P1')['after']['ask'][0]
    after = answer(DAY, {'action': 'link', 'decision': 'same', 'revision': fresh['revision'],
                         'a': candidate['a'], 'b': candidate['b'],
                         'evidence_a': candidate['evidence_a'], 'evidence_b': candidate['evidence_b']},
                   root=edges)
    visit = _visit(after, 'c_main:P1')
    assert visit['nodes'] == ['c_main:G1', 'c_main:P1'] and visit['seconds'] == 300.0
    assert visit['closed'] and visit['after']['total'] == 0


def test_an_answer_about_a_visit_is_retired_when_its_edge_is_repaired(edges):
    fresh = queue(DAY, root=edges)
    visit = _visit(fresh, 'c_main:P1')
    judged = judge(DAY, {'revision': fresh['revision'], 'nodes': visit['nodes'],
                         'evidence': visit['evidence'], 'purity': 'clean', 'wholeness': 'whole'}, root=edges)
    candidate = _visit(judged, 'c_main:P1')['after']['ask'][0]
    after = answer(DAY, {'action': 'link', 'decision': 'same', 'revision': judged['revision'],
                         'a': candidate['a'], 'b': candidate['b'],
                         'evidence_a': candidate['evidence_a'], 'evidence_b': candidate['evidence_b']},
                   root=edges)
    grown = _visit(after, 'c_main:P1')
    assert grown['judgement'] is None and after['counts']['answered'] == 0


def test_a_candidate_the_model_calls_obviously_different_is_listed_not_asked(tmp_path):
    root = tmp_path / 'project'
    _clip(root, 'c_main', '10:00:00', [0, 120], 'P1',
          extra=[(None, [140, 150], (5, 5, 20, 40), (0., 1.))])
    visit = _visit(queue(DAY, root=root), 'c_main:P1')
    assert visit['after']['ask'] == [] and visit['after']['far'] == 1
    assert visit['after']['rest'][0]['distance'] == 1.0
    # Nothing is hidden: the edge counts as closed, and the page says on what grounds.
    assert visit['closed'] and visit['after']['total'] == 1


def test_a_candidate_without_a_descriptor_is_always_asked(tmp_path):
    """A person added by hand on video has no embedding; silence is not evidence."""
    root = tmp_path / 'project'
    _clip(root, 'c_main', '10:00:00', [0, 120], 'P1')
    _clip(root, 'c_next', '10:02:10', [0, 20], 'P9')
    (root / 'data/raw_clips/c_next/emb_osnet_ain_x1_0_msmt17.npz').unlink()
    visit = _visit(queue(DAY, root=root), 'c_main:P1')
    assert [c['node'] for c in visit['after']['ask']] == ['c_next:P9']
    assert visit['after']['ask'][0]['distance'] is None and not visit['closed']


def test_the_queue_ships_crops_not_every_detection_id(day):
    fragment = queue(DAY, root=day)['visits'][0]['fragments'][0]
    assert 'observations' not in fragment and 'ranges' not in fragment
    assert fragment['shots'] and all(len(s['box']) == 4 for s in fragment['shots'])
    assert all(isinstance(v, int) for s in fragment['shots'] for v in s['box'] + [s['frame']])


def test_pointing_at_the_stranger_is_recorded_with_the_mixed_answer(day):
    fresh = queue(DAY, root=day)
    visit = fresh['visits'][0]
    body = {'revision': fresh['revision'], 'nodes': visit['nodes'], 'evidence': visit['evidence'],
            'purity': 'mixed', 'wholeness': 'whole', 'foreign': ['c_late:P1']}
    after = judge(DAY, body, root=day)
    assert after['visits'][0]['judgement']['foreign'] == ['c_late:P1']


@pytest.mark.parametrize('change,message', [
    ({'foreign': ['c_short:P7']}, 'не из этого визита'),
    ({'foreign': ['c_late:P1'], 'purity': 'clean'}, 'только у смешанного'),
])
def test_a_stranger_marked_on_the_wrong_visit_is_refused(day, change, message):
    fresh = queue(DAY, root=day)
    visit = fresh['visits'][0]
    body = {'revision': fresh['revision'], 'nodes': visit['nodes'], 'evidence': visit['evidence'],
            'purity': 'mixed', 'wholeness': 'whole'}
    body.update(change)
    with pytest.raises(ValueError, match=message):
        judge(DAY, body, root=day)


def test_a_brief_answer_returns_the_counters_instead_of_every_visit(day):
    fresh = queue(DAY, root=day)
    visit = fresh['visits'][0]
    after = answer(DAY, {'revision': fresh['revision'], 'nodes': visit['nodes'], 'brief': True,
                         'evidence': visit['evidence'], 'purity': 'clean', 'wholeness': 'whole'}, root=day)
    assert after['brief'] and 'visits' not in after
    assert after['counts']['clean_and_whole'] == 1 and after['revision'] == fresh['revision'] + 1
    assert queue(DAY, root=day)['counts']['answered'] == 1


def _client(root, monkeypatch):
    import label_pieces as web
    monkeypatch.setattr(web, 'ROOT', str(root))
    monkeypatch.setattr(web, 'CLIPS', str(root / 'data/raw_clips'))
    web.app.config['TESTING'] = True
    client = web.app.test_client()
    client.set_cookie('labeler_key', web.KEY)
    return client


def test_the_page_saves_a_boundary_answer_through_its_own_api(edges, monkeypatch):
    client = _client(edges, monkeypatch)
    listing = client.get('/api/day/%s/visits' % DAY).json
    visit = next(v for v in listing['visits'] if 'c_main:P1' in v['nodes'])
    candidate = visit['after']['ask'][0]
    link = {'action': 'link', 'decision': 'different', 'revision': listing['revision'],
            'a': candidate['a'], 'b': candidate['b'], 'brief': True,
            'evidence_a': candidate['evidence_a'], 'evidence_b': candidate['evidence_b']}
    assert client.post('/api/day/%s/visits' % DAY, json={**link, 'revision': 99}).status_code == 409
    saved = client.post('/api/day/%s/visits' % DAY, json=link)
    assert saved.status_code == 200 and saved.json['brief'] and 'visits' not in saved.json
    assert saved.json['counts']['closed'] >= 1
    after = client.get('/api/day/%s/visits' % DAY).json
    assert next(v for v in after['visits'] if 'c_main:P1' in v['nodes'])['closed']
    undone = client.post('/api/day/%s/visits' % DAY,
                         json={'action': 'undo', 'revision': saved.json['revision']})
    assert undone.status_code == 200
    assert not next(v for v in undone.json['visits'] if 'c_main:P1' in v['nodes'])['closed']


def test_an_employee_is_answered_with_one_key_and_kept_out_of_the_acceptance_number(day):
    fresh = queue(DAY, root=day)
    visit = fresh['visits'][0]
    after = judge(DAY, {'revision': fresh['revision'], 'nodes': visit['nodes'],
                        'evidence': visit['evidence'], 'kind': 'staff'}, root=day)
    record = after['visits'][0]['judgement']
    assert record['kind'] == 'staff' and record['purity'] is None and record['wholeness'] is None
    counts = after['counts']
    assert counts['answered'] == 1 and counts['staff'] == 1 and counts['customers_answered'] == 0
    assert counts['clean_and_whole'] == 0


@pytest.mark.parametrize('change,message', [
    ({'kind': 'manager'}, 'Неверный ответ'),
    ({'kind': 'customer', 'purity': None}, 'Неверный ответ'),
    ({'assisted': 'someone'}, 'Неизвестный источник'),
])
def test_who_the_visit_is_and_who_proposed_it_are_checked(day, change, message):
    fresh = queue(DAY, root=day)
    visit = fresh['visits'][0]
    body = {'revision': fresh['revision'], 'nodes': visit['nodes'], 'evidence': visit['evidence'],
            'purity': 'clean', 'wholeness': 'whole'}
    body.update(change)
    with pytest.raises(ValueError, match=message):
        judge(DAY, body, root=day)


def _propose(root, visits=None, links=None):
    atomic_json(root / 'data/day_review_suggestions' / (DAY + '.json'),
                {'source': 'claude', 'visits': visits or {}, 'links': links or {}})


def test_a_proposal_is_shown_on_exactly_the_visit_and_boundary_it_was_made_about(edges):
    fresh = queue(DAY, root=edges)
    visit = _visit(fresh, 'c_main:P1')
    candidate = visit['after']['ask'][0]
    _propose(edges,
             visits={'|'.join(visit['nodes']): {'kind': 'customer', 'purity': 'clean',
                                                'evidence': visit['evidence'], 'confidence': 'high'}},
             links={'%s|%s' % (candidate['a'], candidate['b']): {
                 'decision': 'same', 'evidence_a': candidate['evidence_a'],
                 'evidence_b': candidate['evidence_b'], 'confidence': 'medium'}})
    shown = _visit(queue(DAY, root=edges), 'c_main:P1')
    assert shown['suggestion']['purity'] == 'clean' and shown['suggestion']['by'] == 'claude'
    assert shown['after']['ask'][0]['suggested']['decision'] == 'same'
    assert queue(DAY, root=edges)['counts']['suggested'] == 1


def test_a_proposal_about_something_that_has_since_changed_is_not_shown(edges):
    fresh = queue(DAY, root=edges)
    visit = _visit(fresh, 'c_main:P1')
    candidate = visit['after']['ask'][0]
    _propose(edges,
             visits={'|'.join(visit['nodes']): {'kind': 'staff', 'evidence': [['c_main:P1', 'old']]}},
             links={'%s|%s' % (candidate['a'], candidate['b']): {
                 'decision': 'same', 'evidence_a': 'old', 'evidence_b': candidate['evidence_b']}})
    shown = _visit(queue(DAY, root=edges), 'c_main:P1')
    assert shown['suggestion'] is None and 'suggested' not in shown['after']['ask'][0]


def test_an_answer_given_with_a_proposal_on_screen_says_so(edges):
    fresh = queue(DAY, root=edges)
    visit = _visit(fresh, 'c_main:P1')
    candidate = visit['after']['ask'][0]
    linked = answer(DAY, {'action': 'link', 'decision': 'different', 'revision': fresh['revision'],
                          'a': candidate['a'], 'b': candidate['b'], 'assisted': 'claude',
                          'evidence_a': candidate['evidence_a'], 'evidence_b': candidate['evidence_b']},
                    root=edges)
    judged = judge(DAY, {'revision': linked['revision'], 'nodes': visit['nodes'],
                         'evidence': visit['evidence'], 'purity': 'clean', 'wholeness': 'whole',
                         'assisted': 'claude'}, root=edges)
    stored = read_json(edges / 'data/day_review' / (DAY + '.json'))
    assert stored['links'][0]['assisted'] == 'claude' and stored['visits'][0]['assisted'] == 'claude'
    assert _visit(judged, 'c_main:P1')['judgement']['assisted'] == 'claude'
