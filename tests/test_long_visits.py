"""The long-visit queue: what it shows, and what it refuses to accept as an answer."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'multicam'))
from day_visits import build_day                     # noqa: E402
from long_visits import judge, queue                 # noqa: E402
from storage import atomic_json, read_json           # noqa: E402

DAY = '20260918'


def _clip(root, name, start, times, label, box=(5, 5, 20, 40)):
    d = root / 'data/raw_clips' / name
    d.mkdir(parents=True)
    atomic_json(d / 'meta_yolo26x-seg.json', {'day': DAY, 'seconds': 120,
        'start': '2026-09-18T%s' % start, 'width': 100, 'height': 80})
    atomic_json(d / 'pieces_yolo26x-seg.json', [{'piece': 0, 'cam': 'cam1',
        't0': float(times[0]), 't1': float(times[-1]), 'dets': list(range(len(times)))}])
    atomic_json(d / 'groups_yolo26x-seg.json', [{'person': 0, 'pieces': [0]}])
    atomic_json(d / 'gt_manual.json', {'0': label})
    rows = np.zeros((len(times), 10), np.float32)
    rows[:, 0] = times
    rows[:, 1:5] = box
    np.savez(d / 'dets_yolo26x-seg.npz', cam1=rows, cam2=np.empty((0, 10)))
    np.savez(d / 'emb_osnet_ain_x1_0_msmt17.npz',
             cam1=np.array([[1., 0.]] * len(times)), cam2=np.empty((0, 2)))
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
    assert result['counts'] == {'long_visits': 1, 'answered': 0, 'retired': 0,
                                'clean_and_whole': 0, 'mixed': 0, 'partial': 0, 'fragments': 2}


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


def test_a_fragment_already_in_the_visit_is_never_offered_as_its_continuation(day):
    inside = set(queue(DAY, root=day)['visits'][0]['nodes'])
    offered = {c['node'] for c in queue(DAY, root=day)['visits'][0]['continuations']}
    assert not (offered & inside)


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
