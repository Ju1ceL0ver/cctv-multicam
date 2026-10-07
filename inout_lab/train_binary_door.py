"""Final fit on all 1628 examples with labels_2class.json. Accuracy: see binary2_fixed.py (leave-one-day-out), NOT this fit."""
import json,pickle,numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from binary_door import binary_features
z=np.load('io_cam1.npz');F,X0,G=z['F'],z['X'],z['G'];lab=json.load(open('labels_2class.json'))
X=np.stack([binary_features(F[i],X0[i][:,:,3],G[i,1:]) for i in range(len(G))])
y=np.array([lab[str(i)]=='inside' for i in range(len(X))]).astype(int)
m=ExtraTreesClassifier(300,min_samples_leaf=2,max_features=1.,random_state=0,n_jobs=4).fit(X,y);m.n_jobs=1
pickle.dump(m,open('binary_door.pkl','wb'));np.save('diagnostics/binary_door_X.npy',X);print(X.shape,y.mean())
