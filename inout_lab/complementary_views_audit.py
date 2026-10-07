"""Fixed equal-weight complementary-view blends, diagnostic cached LODO only.
No train-row predictions; incomplete caches are rejected. No test-trained gate.
"""
from pathlib import Path
import json,numpy as np
from owner_boundary_search import metrics
R=Path(__file__).parent
if __name__=='__main__':
 z=np.load(R/'io_cam1.npz');y=z['y'];days=z['day'];v=np.load(R/'diagnostics/original_labels.npy')!=0
 sources={'base':('context_predictions','context_boost'),'aligned':('aligned_density_mlp_predictions',),'mask_geometry':('head_scale_mask_predictions',),'physical_geometry':('calibrated_density_geometry_predictions',),'original_mlp':('expert_mlp_3.0',)}
 P={k:np.mean([np.load(R/f'diagnostics/{n}.npz')['probabilities'] for n in names],axis=0) for k,names in sources.items()}
 for p in P.values():assert p.shape==(1628,3) and np.isfinite(p).all() and np.allclose(p.sum(1),1,atol=1e-4)
 configs=[['base','aligned'],['base','physical_geometry'],['aligned','physical_geometry'],['base','aligned','physical_geometry'],['base','aligned','mask_geometry'],['aligned','mask_geometry'],['base','aligned','physical_geometry','mask_geometry'],list(sources)]
 report={}
 for names in configs:
  key='_'.join(names);p=np.mean([P[n] for n in names],axis=0);report[key]=metrics(p,y,v,days);np.savez(R/f'diagnostics/complementary_{key}.npz',probabilities=p,completed_days=np.unique(days));print(key,report[key]['wrong'],report[key]['gross'],report[key]['gross_indices'])
 (R/'complementary_views_report.json').write_text(json.dumps({'protocol':'Fixed equal weights, previously explored held-out days; all three classes, no abstention','sources':sources,'models':report},indent=2))
