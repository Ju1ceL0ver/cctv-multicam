"""Offline entry/exit matcher constrained to the current inventory.

Each entry starts a visit ID. Repeat visits are deliberately not person-reidentified.
Presence is conditional on observed crossings, never a calibrated probability.

20.10: a second entry clearly matching someone already inside (10 s .. 10 min later) is the door counting the
same person twice -- it continues that visit instead of leaving a ghost in the inventory. Exits a few seconds
apart (a group leaving) are decided together: each exit gets at most one entry, and a link needs a margin over
both the exit's other candidates and the other exits of the group that want the same entry.
"""
import hashlib
import itertools
import numpy as np
from door_identity import similarity

DUP_MIN_S, DUP_MAX_S = 10.0, 600.0      # closer entries are two people walking in together
SAME_TRACK = .85                        # ... unless near-identical (20.10: one person's copies .82-.91, two people <= .69)
GROUP_S = 6.0                           # exits this close are one group
RETURN_S = 300.0                        # an entry this soon after a clear exit of the same person: the visit goes on
SOLE = .55                              # 22.10: the only fitting person inside needs less likeness (owner 1, 21.10: 0.62)
REPEAT_OUT_S = 60.0                     # a near-identical exit this soon after a linked one: the door counted it twice


def _joint(S, threshold):
    """Exits x entries -> entry index or None per exit, maximising the total score above the threshold."""
    n, m = S.shape
    if n * m == 0:
        return [None] * n
    if n > 3 or m > 12:                 # too many for brute force: each exit's own best
        return [int(np.argmax(r)) if r.max() >= threshold else None for r in S]
    best, pick = -1.0, [None] * n
    for perm in itertools.product(*[[None] + [j for j in range(m) if S[i, j] >= threshold] for i in range(n)]):
        used = [p for p in perm if p is not None]
        if len(used) != len(set(used)):
            continue
        total = sum(S[i, p] for i, p in enumerate(perm) if p is not None)
        if total > best:
            best, pick = total, list(perm)
    return pick


def _fits(a,b):
    """Roles 'staff' / 'customer' (from door_role) keep two inventories apart; an unknown role fits both."""
    return a.get('role') is None or b.get('role') is None or a['role']==b['role']


def match(events,vectors,threshold=.65,margin=.06,force_best=False):
    inside={};output={};seen=set();order=[];left={}     # left: entry key -> person who walked out recently
    ev=[];
    for e in sorted(events,key=lambda e:(e['t'],e['key'])):
        if e['key'] in seen or e['kind'] not in ('in','out'):continue
        seen.add(e['key']);ev.append(e)
    t_of={e['key']:e['t'] for e in ev}
    i=0
    while i<len(ev):
        e=ev[i];key=e['key']
        if e['kind']=='in':
            pid='V'+hashlib.sha256(key.encode()).hexdigest()[:16];status='entry';v=vectors.get(key)
            if v is not None and not force_best:
                for k in [k for k,p in left.items() if e['t']-p['out_t']>RETURN_S]:del left[k]
                pool=dict(left,**inside)
                c=sorted(((similarity(v,p['vector']),k) for k,p in pool.items()
                          if p['vector'] is not None and e['t']-p['event']['t']<=DUP_MAX_S+RETURN_S and _fits(e,p['event'])),reverse=True)
                same=lambda q:q[0]>=SAME_TRACK
                ok=lambda q:same(q) or q[0]>=threshold and (q[1] in left or e['t']-pool[q[1]]['event']['t']>=DUP_MIN_S)
                c=[q for q in c if ok(q)] if c and ok(c[0]) else []
                if c and (len(c)<2 or c[0][0]-c[1][0]>=margin or same(c[0])):
                    back=c[0][1] in left
                    pid=(left if back else inside).pop(c[0][1])['id'];status='return' if back else 'repeat_entry'
            inside[key]=dict(id=pid,event=e,vector=v,uncertain=False)
            output[key]=dict(event=key,person=pid,entry_event=key,status=status,candidates=[]);order.append(key)
            i+=1;continue
        group=[e];j=i+1
        while j<len(ev) and ev[j]['kind']=='out' and ev[j]['t']-group[-1]['t']<=GROUP_S:group.append(ev[j]);j+=1
        # 20.10 evening: one person's exit counted twice took the same entry from itself -- both stayed unlinked
        repeat={};primary=[]
        for x in group:
            v=vectors.get(x['key']);twin=None
            if v is not None and not force_best:
                for y in primary:
                    if vectors.get(y['key']) is not None and similarity(v,vectors[y['key']])>=SAME_TRACK:twin=('group',y['key']);break
                if twin is None:
                    for k,p in left.items():
                        if p.get('out_vector') is not None and x['t']-p['out_t']<=REPEAT_OUT_S and _fits(x,p['event']) \
                                and similarity(v,p['out_vector'])>=SAME_TRACK:twin=('left',k);break
            if twin:repeat[x['key']]=twin
            else:primary.append(x)
        entries=[k for k,p in inside.items() if p['vector'] is not None]
        ranks={}
        for x in group:
            v=vectors.get(x['key']);rank=[]
            if v is not None:
                for k in entries:
                    p=inside[k]
                    if not _fits(x,p['event']):continue
                    rank.append(dict(entry_event=k,person=p['id'],score=similarity(v,p['vector']),
                                     presence='uncertain' if p['uncertain'] else 'observed_entry'))
            rank.sort(key=lambda r:(-r['score'],r['entry_event']));ranks[x['key']]=rank
        # 20.10: an exit more like someone who has just walked out than like anyone inside is the door counting that
        # exit again (a person lingering in the doorway), not a stranger -- it must not take an open entry
        for x in list(primary):
            v=vectors.get(x['key'])
            if v is None or force_best:continue
            best_in=ranks[x['key']][0]['score'] if ranks[x['key']] else -1.
            near=[(max(similarity(v,p['vector']) if p['vector'] is not None else -1.,
                       similarity(v,p['out_vector']) if p.get('out_vector') is not None else -1.),k)
                  for k,p in left.items() if x['t']-p['out_t']<=RETURN_S and _fits(x,p['event'])]
            if near and max(near)[0]>=threshold and max(near)[0]>best_in:
                repeat[x['key']]=('left',max(near)[1]);primary.remove(x)
        live=[x for x in primary if ranks[x['key']]]
        S=np.array([[next((r['score'] for r in ranks[x['key']] if r['entry_event']==k),-1.) for k in entries] for x in live]).reshape(len(live),len(entries))
        pick=_joint(S,-.99 if force_best else threshold)   # -1: another role, never
        # 22.10: two open entries of one person (a missed exit in between, 0.74/0.77 on staff 2) are no rivals
        twin=lambda j,l:similarity(inside[entries[j]]['vector'],inside[entries[l]]['vector'])>=SAME_TRACK
        chosen={}
        for a,x in enumerate(live):
            fit=[j for j in range(len(entries)) if S[a,j]>-1]
            if pick[a] is None and not force_best and fit:
                b=max(fit,key=lambda j:S[a,j])
                if S[a,b]>=SOLE and all(twin(b,j) for j in fit if j!=b) and b not in pick:
                    pick[a]=b                            # the only person inside it can be
        for a,(x,p) in enumerate(zip(live,pick)):
            if p is None:continue
            others=[S[a,j] for j in range(len(entries)) if j!=p and not twin(p,j)]
            rival=max(others+list(np.delete(S[:,p],a))+[-1.])
            if force_best or S[a,p]-rival>=margin:chosen[x['key']]=entries[p]
        for x in primary:
            rank=ranks[x['key']];k=chosen.get(x['key'])
            gap=rank[0]['score']-rank[1]['score'] if len(rank)>1 else None
            if k is not None:
                p=inside.pop(k);pid=p['id'];left[k]=dict(p,out_t=x['t'],out_vector=vectors.get(x['key']))
                for tw in [t for t,q in inside.items() if q['vector'] is not None and p['vector'] is not None and
                           similarity(q['vector'],p['vector'])>=SAME_TRACK]:
                    inside.pop(tw)                       # its twin entries leave with it
            else:pid=None
            output[x['key']]=dict(event=x['key'],person=pid,entry_event=k,
                                  status='forced_hypothesis' if k and force_best else 'matched' if k else 'pending',
                                  margin=gap,candidates=rank);order.append(x['key'])
        for x in group:
            if x['key'] not in repeat:continue
            where,k=repeat[x['key']]
            src=output[k] if where=='group' else dict(person=left[k]['id'],entry_event=k)
            output[x['key']]=dict(event=x['key'],person=src['person'],entry_event=src['entry_event'],
                                  status='repeat_exit' if src['person'] else 'pending',margin=None,candidates=ranks[x['key']])
            order.append(x['key'])
        order[-len(group):]=sorted(order[-len(group):],key=lambda k:(t_of[k],k))
        if any(chosen.get(x['key']) is None for x in primary):
            # An unidentified exit invalidates certainty about every remaining candidate.
            for p in inside.values():p['uncertain']=True
        i=j
    return [output[k] for k in order]
