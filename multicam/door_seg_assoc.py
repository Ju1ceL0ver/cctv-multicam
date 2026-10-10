"""Experimental BYTE recovery for weak masks: weak detections never create IDs."""
import numpy as np
from scipy.optimize import linear_sum_assignment
from door_seg_track import Tracker,Track,centre,shifted_iou


class ScoreTracker(Tracker):
    def __init__(self, config=None, birth_score=.4, recover_score=.1, mask_gate=.5):
        super().__init__(config)
        self.birth_score,self.recover_score,self.mask_gate=birth_score,recover_score,mask_gate

    def update(self,tick,predictions):
        if self.tick is not None and tick<=self.tick:raise ValueError('ticks must increase')
        self.tick=tick;c=self.config
        self.tracks={pid:t for pid,t in self.tracks.items() if tick-t.tick<=c.max_gap}
        old=list(self.tracks.values());p=[z for z in predictions if z[2]>=self.recover_score]
        costs=np.array([[self.cost(tr,z,tick) for z in p] for tr in old],float).reshape(len(old),len(p))
        linked={};used=set()
        for weak,adjacent in ((False,True),(False,False),(True,True),(True,False)):
            rows=[i for i,tr in enumerate(old) if i not in used and (tick-tr.tick==1)==adjacent]
            cols=[j for j,z in enumerate(p) if j not in linked and (z[2]<self.birth_score)==weak]
            if not rows or not cols:continue
            sub=costs[np.ix_(rows,cols)].copy()
            if weak:
                for ii,i in enumerate(rows):
                    for jj,j in enumerate(cols):
                        gap=tick-old[i].tick
                        delta=old[i].pred[4] if gap==1 else old[i].velocity*gap
                        if shifted_iou(old[i].pred[0],p[j][0],delta)<self.mask_gate:sub[ii,jj]=np.inf
            allowed=sub<=c.max_cost
            for i,j in zip(*np.nonzero(allowed)):
                rival=min(np.min(np.delete(sub[i],j),initial=np.inf),np.min(np.delete(sub[:,j],i),initial=np.inf))
                if rival<=c.max_cost and rival-sub[i,j]<c.ambiguity:allowed[i,j]=False;self.stats['ambiguous']+=1
            mat=np.full((len(rows),len(cols)+len(rows)),1e6);mat[:,:len(cols)]=np.where(allowed,sub,1e6)
            for i in range(len(rows)):mat[i,len(cols)+i]=c.max_cost+1e-6
            ii,jj=linear_sum_assignment(mat)
            for i,j in zip(ii,jj):
                if j<len(cols) and allowed[i,j]:linked[cols[j]]=old[rows[i]];used.add(rows[i])
        out=[]
        for j,z in enumerate(p):
            if j in linked:
                tr=linked[j];gap=tick-tr.tick;v=(centre(z[1])-centre(tr.pred[1]))/gap
                speed=np.linalg.norm(v)
                if speed>c.max_step:v*=c.max_step/speed
                pid=tr.pid;self.stats['adjacent' if gap==1 else 'recovered']+=1
            elif z[2]>=self.birth_score:
                pid=self.next_id;self.next_id+=1;v=np.zeros(2);self.stats['births']+=1
            else:continue
            self.tracks[pid]=Track(pid,tick,z,v);out.append((pid,z))
        return out
