"""Fixed equal-view comparisons after complete LODO reports exist.
No fit to held-day labels and no row-specific rules. Diagnostic selection only.
"""
from pathlib import Path
import numpy as np,json
from owner_boundary_search import metrics
R=Path(__file__).parent
if __name__=='__main__':
 z=np.load(R/'io_cam1.npz');y=z['y'];days=z['day'];v=np.load(R/'diagnostics/original_labels.npy')!=0
 names={'owner':'owner_boundary_mlp_equal','old':'expert_mlp_3.0','base1':'context_predictions','base2':'context_boost','track':'owner_track_context_30','body':'frozen_body_boost','physical':'calibrated_density_geometry_predictions'}
 P={k:np.load(R/f'diagnostics/{name}.npz')['probabilities'] for k,name in names.items()}
 for p in P.values():assert p.shape==(1628,3) and np.isfinite(p).all() and np.allclose(p.sum(1),1,atol=1e-4)
 P['base']=(P.pop('base1')+P.pop('base2'))/2
 configs=[['owner','base','track'],['owner','old','base','track'],['owner','old','base','track','physical'],['owner','old'],['owner','base'],['owner','old','base'],['owner','track'],['owner','old','track'],['owner','body'],['owner','old','body'],['owner','base','body']]
 out={}
 for members in configs:
  name='_'.join(members);p=np.mean([P[k] for k in members],axis=0);out[name]=metrics(p,y,v,days);np.savez(R/f'diagnostics/new_views_{name}.npz',probabilities=p,completed_days=np.unique(days));print(name,out[name]['wrong'],out[name]['gross'],out[name]['gross_indices'],flush=True)
 (R/'owner_experiment_blends_report.json').write_text(json.dumps({'models':out,'sources':names,'protocol':'Fixed equal weights, no abstention; all days; diagnostic comparisons not fresh blind test'},indent=2))
