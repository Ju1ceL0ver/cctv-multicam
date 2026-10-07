from pathlib import Path
import numpy as np,pickle,json
from sklearn.ensemble import ExtraTreesClassifier
root=Path(__file__).parent;z=np.load(root/'io_cam1.npz');y=z['y'];v=np.load(root/'diagnostics/original_labels.npy')!=0;A=np.column_stack([np.load(root/('diagnostics/'+n+'_features.npy')) for n in ['rich','head','context']]+[np.load(root/'diagnostics/crop_rgb_features.npy')]);m=ExtraTreesClassifier(n_estimators=200,min_samples_leaf=1,max_features=.7,random_state=19,n_jobs=3).fit(A[v],y[v]);m.n_jobs=1;path=root/'door_crop_classifier.pkl';path.write_bytes(pickle.dumps(m,protocol=5));print('saved',path.name,path.stat().st_size,flush=True)
