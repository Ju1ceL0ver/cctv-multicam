"""Reference-supported head expert for truncated/inconsistent body metadata.
Trust thresholds come only from visible-foot references on training days.
No test labels used to choose the expert; all classes remain outputs.
"""
from pathlib import Path
import json,numpy as np
from calibrated_density_search import CameraScale

ROOT=Path(__file__).parent
if __name__=='__main__':
    z=np.load(ROOT/'io_cam1.npz');y=z['y'];days=z['day'];valid=np.load(ROOT/'diagnostics/original_labels.npy')!=0
    A=np.load(ROOT/'diagnostics/aligned_features.npy')[:,:1584];G=np.load(ROOT/'diagnostics/aligned_geometry.npy');cal=json.loads((ROOT/'diagnostics/calib_final_source.json').read_text())['cam1']
    scale=CameraScale(cal,A[:,234:936],G,A[:,:234],np.load(ROOT/'diagnostics/pose_features.npy'),np.load(ROOT/'diagnostics/pose_512_features.npy'))
    base=np.load(ROOT/'diagnostics/expert_mlp_3.0.npz')['probabilities'];head=np.load(ROOT/'diagnostics/calibrated_density_geometry_predictions.npz')['probabilities'];P=base.copy();audit={}
    discrepancy=np.max(np.abs((G[:,:4]-z['G'][:,1:5])*[1280,720,1280,720]),axis=1)
    for d in np.unique(days):
        tr=np.where(valid&(days!=d))[0];te=days==d;S,_=scale.transform(tr)
        # Each proposal stores q25/q50/q75 [log-density,area,height,z,footx,footy].
        body=np.stack([S[:,j*18+8] for j in range(4)],1)
        zhead=np.stack([S[:,j*18+9] for j in range(4)],1)
        height=np.median(body,1);hz=np.median(zhead,1)
        spread=np.std(zhead,1)
        visible=scale.visible[tr]&(height[tr]>0)&(hz[tr]>.1)&(hz[tr]<.7)
        refs=tr[visible]
        assert len(refs)>=20
        hlow=float(np.quantile(height[refs],.1));zlow,zhigh=np.quantile(hz[refs],[.05,.95]);maxspread=float(np.quantile(spread[refs],.95));bbox_noise=max(30.,float(np.quantile(discrepancy[refs],.95)))
        uncertain_body=(height<hlow)|(discrepancy>bbox_noise)
        supported=(hz>=zlow)&(hz<=zhigh)&(spread<=maxspread)
        choose=te&uncertain_body&supported&(head.max(1)>=.5)
        P[choose]=head[choose]
        audit[str(d)]={'references':len(refs),'body_height_min':hlow,'head_height_range':[float(zlow),float(zhigh)],'head_spread_max':maxspread,'bbox_noise_max':bbox_noise,'selected_indices':np.where(choose)[0].tolist()}
    pred=P.argmax(1);gross=valid&(y<2)&(pred<2)&(pred!=y);old=base.argmax(1);oldgross=valid&(y<2)&(old<2)&(old!=y)
    report={'defined':int(valid.sum()),'accuracy':float((pred[valid]==y[valid]).mean()),'wrong':int((valid&(pred!=y)).sum()),'gross':int(gross.sum()),'gross_indices':np.where(gross)[0].tolist(),'fixed_gross':np.where(oldgross&~gross)[0].tolist(),'new_gross':np.where(gross&~oldgross)[0].tolist(),'doorway_recall':float((pred[y==2]==2).mean()),'audit':audit}
    np.savez(ROOT/'diagnostics/physical_scale_gate_predictions.npz',probabilities=P);(ROOT/'physical_scale_gate_report.json').write_text(json.dumps(report,indent=2));print({k:v for k,v in report.items() if k!='audit'},flush=True)
