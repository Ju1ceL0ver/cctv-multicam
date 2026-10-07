"""Verify original provenance; regenerate target crop features from source pixels."""
from pathlib import Path
import hashlib,json,time,zipfile,shutil,tempfile
import numpy as np,cv2
from door_features import expanded_features
from crop_feature_search import crop_features
R=Path(__file__).parent
if __name__=='__main__':
 start=time.perf_counter();z=np.load(R/'io_cam1.npz');q=np.load(R/'diagnostics/recovered_crops.npz');report=json.loads((R/'diagnostics/recovery_report.json').read_text());assert not report['source_replay_mismatches'];assert report['ids_sha256']==hashlib.sha256(json.dumps(z['ids'].tolist()).encode()).hexdigest();assert report['labels_sha256']==hashlib.sha256(z['y'].tobytes()).hexdigest()
 for key in ['ids','y','day']:assert np.array_equal(q[key],z[key]),key
 G=q['G'][:,1:];assert np.allclose(G,np.load(R/'diagnostics/aligned_geometry.npy'),atol=1e-7);assert np.array_equal(q['G'][:,7:9],z['G'][:,7:9])
 with tempfile.TemporaryDirectory(dir=R/'diagnostics',prefix='recovered_') as tmp:
  with zipfile.ZipFile(R/'io_cam1.npz') as a,a.open('F.npy') as src,open(Path(tmp)/'F.npy','wb') as dst:shutil.copyfileobj(src,dst)
  F=np.load(Path(tmp)/'F.npy',mmap_mode='r');X=q['X'];rows=[]
  for i,(f,x,g) in enumerate(zip(F,X,G)):
   rows.append(np.concatenate([expanded_features(f,x[:,:,3],g),crop_features(x)]))
   if i in [393,970,1149,1394]:cv2.imwrite(str(R/f'diagnostics/recovered_{i}_crop.jpg'),x[:,:,:3][:,:,::-1])
   if (i+1)%256==0:print('recovered features',i+1,'elapsed',round(time.perf_counter()-start),flush=True)
  B=np.asarray(rows,np.float32);assert B.shape==(1628,3288) and np.isfinite(B).all();np.save(R/'diagnostics/recovered_features.npy',B);del F
 old=np.load(R/'diagnostics/aligned_features.npy');out={'rows':len(B),'same_ids_labels_days':True,'same_geometry':True,'same_foot_coordinates':True,'remote_full_frame_replay_identical':True,'feature_rows_changed':int((np.abs(B-old).max(1)>1e-6).sum()),'seconds':time.perf_counter()-start};(R/'recovered_crop_features_report.json').write_text(json.dumps(out,indent=2));print(out,flush=True)
