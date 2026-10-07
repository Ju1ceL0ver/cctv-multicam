def _export_inout_checks_0710():
 import sys,json,cv2,numpy as np
 from pathlib import Path
 root=Path(r'C:\Users\ArykovAA\cctv_ai\multicam');sys.path.insert(0,str(root));import rate
 z=np.load(root/'data/inout/io_cam1.npz');samples=json.loads((root/'data/inout/samples.json').read_text(encoding='utf-8'));frames=rate.index(root);out=root/'data/inout_lab_source_recovery'
 for i in [393,970,1149,1394]:
  s=samples[str(z['ids'][i])];img=cv2.imread(str(frames[s['frame']][0]));img=cv2.resize(img,(1280,720)) if img.shape[:2]!=(720,1280) else img;lab=cv2.imread(str(frames[s['frame']][1]),cv2.IMREAD_UNCHANGED)
  if lab.ndim==3:lab=lab[:,:,0]
  if lab.shape!=(720,1280):lab=cv2.resize(lab,(1280,720),interpolation=cv2.INTER_NEAREST)
  cv2.imwrite(str(out/f'api_{i}_rgb.png'),img);cv2.imwrite(str(out/f'api_{i}_mask.png'),(lab==s['value']).astype('uint8')*255)
 print('four original RGB/target-mask pairs saved in experimental directory')
_export_inout_checks_0710()
