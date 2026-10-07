"""Independent checkpoint replay, three-class error audit and single-target timing.
Consumes only locally trained weights. No fitting or thresholds selected on test.
"""
from pathlib import Path
import numpy as np,json,time
import torch
from torch.nn import functional as F
from dual_view_cnn import DualViewNet
from spatial_cnn import cached_frames
root=Path(__file__).parent

def audit(day='20260919'):
 path=root/('diagnostics/dualview_'+day);saved=np.load(str(path)+'_predictions.npz');ix=saved['indices'];prob=saved['probabilities'];z=np.load(root/'io_cam1.npz');y=z['y'];days=z['day'];valid=np.load(root/'diagnostics/original_labels.npy')!=0
 assert np.array_equal(ix,np.where(valid&(days==day))[0]),'Incomplete held-day coverage'
 state=torch.load(str(path)+'.pt',map_location='cpu',weights_only=False);assert state['held_day']==day
 device='mps' if torch.backends.mps.is_available() else 'cpu';model=DualViewNet().to(device);model.load_state_dict(state['state_dict']);model.eval();frames=cached_frames(z);crop=z['X'];G=(z['G'][:,1:]-state['geometry_mean'])/state['geometry_std']
 def inputs(ids):
  f=torch.from_numpy(np.array(frames[ids],copy=True)).to(device);c=torch.from_numpy(crop[ids].transpose(0,3,1,2).copy()).float().to(device);c=F.interpolate(c,size=(88,160),mode='bilinear',align_corners=False);g=torch.from_numpy(G[ids].astype(np.float32)).to(device);return f,c,g
 replay=[]
 with torch.inference_mode():
  for ids in np.array_split(ix,max(1,int(np.ceil(len(ix)/24)))):replay.append(model(*inputs(ids)).softmax(1).cpu().numpy())
  sample=inputs(ix[:1]);model(*sample);torch.mps.synchronize() if device=='mps' else None;start=time.perf_counter()
  for _ in range(30):model(*sample)
  torch.mps.synchronize() if device=='mps' else None;ms=(time.perf_counter()-start)/30*1000
 replay=np.concatenate(replay);difference=float(np.abs(replay-prob).max());assert difference<1e-5,difference
 pred=prob.argmax(1);confusion=np.zeros((3,3),int);np.add.at(confusion,(y[ix],pred),1);gross=(y[ix]<2)&(pred<2)&(pred!=y[ix]);report={'day':day,'defined':len(ix),'accuracy':float((pred==y[ix]).mean()),'wrong':int((pred!=y[ix]).sum()),'gross':int(gross.sum()),'gross_indices':ix[gross].tolist(),'confusion_rows_true_columns_pred':confusion.tolist(),'class_order':['outside','inside','doorway'],'replay_max_difference':difference,'prepared_single_target_ms':ms,'timing_includes':'both convolution views and geometry head; excludes decoding, segmentation and input preparation','weight_bytes':Path(str(path)+'.pt').stat().st_size,'device':device}
 # Existing baseline is also day-excluded; no refitting/relabeling.
 base=(np.load(root/'diagnostics/context_predictions.npz')['probabilities']+np.load(root/'diagnostics/context_boost.npz')['probabilities'])/2;bp=base[ix].argmax(1);report['baseline_wrong']=int((bp!=y[ix]).sum());report['baseline_gross']=int(((y[ix]<2)&(bp<2)&(bp!=y[ix])).sum());report['fixed_indices']=ix[(bp!=y[ix])&(pred==y[ix])].tolist();report['regressed_indices']=ix[(bp==y[ix])&(pred!=y[ix])].tolist();Path(str(path)+'_audit.json').write_text(json.dumps(report,indent=2));print('DUAL_VIEW_AUDIT',report,flush=True)
 return report
if __name__=='__main__':audit()
