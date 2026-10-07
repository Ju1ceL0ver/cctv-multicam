"""Independent replay of saved held-day MLP checkpoints and coverage audit."""
from pathlib import Path
import argparse,json,time
import numpy as np,torch
from head_scale_mlp_search import ScaleExperts
from owner_boundary_mlp import make_parts
R=Path(__file__).parent
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--tag',default='');ap.add_argument('--features-source');ap.add_argument('--body-features');ap.add_argument('--pose-features');ap.add_argument('--seed',type=int,default=19);args=ap.parse_args();suffix=('_'+args.tag) if args.tag else '';torch.set_num_threads(1)
 z=np.load(R/'io_cam1.npz');parts=make_parts(args.features_source,args.body_features,args.pose_features);saved=np.load(R/f'diagnostics/owner_boundary_mlp{suffix}_{args.seed}.npz');P=saved['probabilities'];days=z['day'];valid=np.load(R/'diagnostics/original_labels.npy')!=0;assert np.isfinite(P).all();assert set(map(str,np.unique(days)))==set(saved['completed_days'].tolist());maxerr=0;records=[]
 for d in np.unique(days):
  cp=torch.load(R/f'diagnostics/owner_boundary_mlp{suffix}_{args.seed}_{d}.pt',map_location='cpu',weights_only=False);assert str(d)==cp['excluded_day'];assert str(d) not in set(map(str,cp['training_days']));assert set(map(str,cp['training_days']))==set(map(str,np.unique(days[days!=d])))
  model=ScaleExperts(cp['dims']).eval();model.load_state_dict(cp['state_dict']);te=days==d;X=[torch.from_numpy(((p[te]-m)/s).astype('float32')) for p,m,s in zip(parts,cp['means'],cp['stds'])]
  with torch.inference_mode():pred=model(*X)[0].softmax(1).numpy()
  err=float(np.max(abs(pred-P[te])));maxerr=max(maxerr,err);assert err<2e-4;records.append({'day':str(d),'rows':int(te.sum()),'defined':int((te&valid).sum()),'max_probability_error_cpu_vs_saved':err})
 report={'max_probability_error':maxerr,'whole_day_exclusion_verified':True,'defined':int(valid.sum()),'doorway_retained':int((valid&(z['y']==2)).sum()),'folds':records,'parameters':sum(t.numel() for t in model.parameters())};(R/f'owner_boundary_mlp{suffix}_audit.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
