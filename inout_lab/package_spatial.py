from pathlib import Path
import numpy as np,pickle,json
from sklearn.ensemble import ExtraTreesClassifier
from spatial_prior import SpatialPrior,SpatialFeatureClassifier
root=Path(__file__).parent;z=np.load(root/'io_cam1.npz');y=z['y'];days=z['day'];valid=np.load(root/'diagnostics/original_labels.npy')!=0;A=np.column_stack([np.load(root/('diagnostics/'+n+'_features.npy')) for n in ['rich','head','context']]);prior=SpatialPrior().fit(A[valid],y[valid]);spatial=np.empty((valid.sum(),144),np.float32);ix=np.where(valid)[0]
for day in np.unique(days):
 held=days[ix]==day;fit=valid&(days!=day);spatial[held]=SpatialPrior().fit(A[fit],y[fit]).transform(A[ix[held]])
for name,limit in [('means',36),('means_knn',144)]:
 forest=ExtraTreesClassifier(n_estimators=160,min_samples_leaf=1,max_features=.7,random_state=19,n_jobs=1).fit(np.column_stack([A[valid],spatial[:,:limit]]),y[valid]);model=SpatialFeatureClassifier(prior,forest,limit);path=root/('door_spatial_'+name+'.pkl');path.write_bytes(pickle.dumps(model,protocol=5));print(path.name,path.stat().st_size,flush=True)
(root/'spatial_fit.json').write_text(json.dumps({'n':int(valid.sum()),'feature_training':'each training row receives prior fitted on OTHER days','held_day_evaluation':'spatial_report.json; each outer validation fold excludes held day also from every inner fit','default_production_model':'door_classifier.pkl is unchanged'},indent=2))
