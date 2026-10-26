"""Identity safety checks: missing/ambiguous views must not invent a match."""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'multicam'))
from door_identity import assign, metrics


def event(key,t,kind,w=1):
    return dict(key=key,t=t,kind=kind,w=w,answer_key=key)


def test_exit_keeps_entry_id_across_tracker_restart():
    events=[event('a',0,'in',90),event('b',600,'out',1)]
    vectors={k:np.array([[1.,0.]]) for k in ('a','b')}
    result=assign(events,vectors)
    assert result[0]['person']==result[1]['person']=='P0001'
    assert result[1]['entry_event']=='a'


def test_ambiguous_exit_stays_pending():
    events=[event('a',0,'in',1),event('b',.1,'in',2),event('c',600,'out',3)]
    vectors={k:np.array([[1.,0.]]) for k in ('a','b','c')}
    result=assign(events,vectors)
    assert result[0]['person']!=result[1]['person']
    assert result[2]['person'] is None
    assert result[2]['entry_event'] is None


def test_no_images_no_exit_match():
    result=assign([event('a',0,'in'),event('b',600,'out')],{})
    assert result[0]['status']=='missing_views'
    assert result[1]['person'] is None


def test_metrics_count_wrong_identity_even_with_valid_crossings():
    events=[event('a',0,'in'),event('b',600,'out')]
    result=assign(events,{k:np.array([[1.,0.]]) for k in ('a','b')})
    score=metrics(events,result,{'a':1,'b':2},{'a','b'})
    assert score['wrong']==1 and score['mixed_person_ids']==1


def test_inventory_never_reuses_departed_id_for_new_entry():
    from door_presence import match
    events=[event('a',0,'in'),event('b',10,'out'),event('c',400,'in'),event('d',410,'out')]   # past RETURN_S
    result=match(events,{k:np.array([[1.,0.]]) for k in 'abcd'})
    assert result[0]['person']==result[1]['person']
    assert result[2]['person']==result[3]['person']
    assert result[0]['person']!=result[2]['person']
    assert [a['entry_event'] for a in result[3]['candidates']]==['c']


def test_unresolved_exit_marks_inventory_uncertain_and_forcing_is_explicit():
    from door_presence import match
    events=[event('a',0,'in',1),event('b',2,'in',2),event('c',10,'out'),event('d',20,'out')]
    alike=np.array([[.8,.6]])                   # two look-alikes: 0.8 to each other, below SAME_TRACK
    vectors={'a':np.array([[1.,0.]]),'b':alike,'c':np.array([[.95,.312]]),'d':np.array([[.95,.312]])}
    result=match(events,vectors)
    assert result[2]['status']=='pending'
    assert all(c['presence']=='uncertain' for c in result[3]['candidates'])
    forced=match(events,vectors,force_best=True)
    assert forced[2]['status']=='forced_hypothesis'
    assert len(forced[3]['candidates'])==1


def test_inventory_ids_survive_insertion_of_unrelated_earlier_entry():
    from door_presence import match
    events=[event('a',10,'in'),event('b',20,'out')]
    vectors={'a':np.array([[1.,0.]]),'b':np.array([[1.,0.]]),'x':np.array([[0.,1.]])}
    original=match(events,vectors)
    changed=match([event('x',0,'in')]+events,vectors)
    assert original[0]['person']==changed[1]['person']


def test_double_entry_of_one_person_is_one_visit():
    from door_presence import match
    events=[event('a',0,'in'),event('b',30,'in'),event('c',90,'out')]
    result=match(events,{k:np.array([[1.,0.]]) for k in 'abc'})
    assert result[1]['status']=='repeat_entry' and result[1]['person']==result[0]['person']
    assert result[2]['entry_event']=='b' and len(result[2]['candidates'])==1


def test_group_exit_is_decided_jointly():
    from door_presence import match
    a,b=np.array([[1.,0.]]),np.array([[0.,1.]])
    x=np.array([[.8,.6]])                       # 0.8 to a, 0.6 to b
    events=[event('a',0,'in'),event('b',1,'in'),event('x',100,'out'),event('y',102,'out')]
    result=match(events,{'a':a,'b':b,'x':x,'y':a},threshold=.5,margin=.05)
    assert result[3]['entry_event']=='a'         # y is clearly a
    assert result[2]['entry_event']!='a'         # so x cannot also take a


def test_same_track_counted_twice_is_one_visit_even_when_close():
    from door_presence import match
    events=[event('a',0,'in',7),event('b',3,'in',7),event('c',90,'out',9)]
    result=match(events,{k:np.array([[1.,0.]]) for k in 'abc'})
    assert result[1]['status']=='repeat_entry' and result[2]['entry_event']=='b'


def test_staff_exit_never_takes_a_customer_entry():
    from door_presence import match
    events=[dict(event('a',0,'in'),role='customer'),dict(event('b',60,'out',2),role='staff')]
    result=match(events,{k:np.array([[1.,0.]]) for k in 'ab'})
    assert result[1]['entry_event'] is None and result[1]['candidates']==[]


def test_return_within_minutes_keeps_the_visit_id():
    from door_presence import match
    events=[event('a',0,'in',1),event('b',60,'out',2),event('c',120,'in',3),event('d',300,'out',4)]
    result=match(events,{k:np.array([[1.,0.]]) for k in 'abcd'})
    assert result[2]['status']=='return' and len({r['person'] for r in result})==1


def test_exit_counted_twice_does_not_block_itself():
    from door_presence import match
    events=[event('a',0,'in',1),event('b',100,'out',2),event('c',101,'out',3)]
    result=match(events,{k:np.array([[1.,0.]]) for k in 'abc'})
    assert result[1]['entry_event']=='a' and result[2]['status']=='repeat_exit'
    assert result[1]['person']==result[2]['person']==result[0]['person']
