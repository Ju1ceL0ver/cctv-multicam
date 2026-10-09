"""Replay cached train predictions against a saved original tracker, without GPU."""
import json,time,torch
from pathlib import Path
import door_seg as S
import argparse,importlib.util
parser=argparse.ArgumentParser();parser.add_argument("--baseline",required=True);args=parser.parse_args()
spec=importlib.util.spec_from_file_location("tracker_baseline",args.baseline);module=importlib.util.module_from_spec(spec)
import sys;sys.modules[spec.name]=module;spec.loader.exec_module(module);Old=module.Tracker
from door_seg_track import Tracker as New
r=S.ROOT/'data/quality_backups/doorseg_tracker_diag';saved=torch.load(r/'train_predictions.pt',weights_only=False)
results={}
for name,cls in [('original',Old),('cascade',New)]:
 tr=cls();prev={};correct=wrong=broken=total=0;t=time.perf_counter()
 for tick,p,gt,_ in saved:
  linked=tr.update(tick,p)
  matches=S._match([pp[0] for pp in p],[m for m,b,w in gt]);now={gt[j][2]:linked[i][0] for i,j,_ in matches}
  # matched consecutive teacher identity links: breaks vs contamination.
  for who,pid in now.items():
   if who in prev:
    total+=1
    if prev[who]==pid:correct+=1
    else:broken+=1
   if any(other!=who and oldpid==pid for other,oldpid in prev.items()):wrong+=1
  prev=now
 results[name]={'stats':tr.stats,'true_adjacent_links':total,'continued':correct,'broken':broken,'cross_identity_links':wrong,'replay_ms_per_tick':(time.perf_counter()-t)*1000/len(saved)}
(r/'replay.json').write_text(json.dumps(results,indent=2));print(results,flush=True)
