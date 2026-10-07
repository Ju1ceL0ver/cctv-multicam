"""CPU-only re-crop from original images; strict replay of frozen NPZ before use.
Only writes new experimental data/inout_lab_source_recovery files.
"""
def _recover_inout_crops_0710():
    import sys,json,time,hashlib,math
    from pathlib import Path
    import numpy as np,cv2
    root=Path(r'C:\Users\ArykovAA\cctv_ai\multicam');sys.path.insert(0,str(root));import rate,rooms,io_fast as I
    z=np.load(root/'data/inout/io_cam1.npz');ids=z['ids'];oldF=z['F'];G=z['G'].copy();samples=json.loads((root/'data/inout/samples.json').read_text(encoding='utf-8'));frames=rate.index(root)
    floor=cv2.resize(rooms.mask('cam1'),(1280,720),interpolation=cv2.INTER_NEAREST)>0;gx,gy=np.meshgrid(np.linspace(0,255,1280),np.linspace(0,255,720));crops=[];bad=[];start=time.time()
    for i,sid in enumerate(ids):
        sample=samples[str(sid)];img=cv2.imread(str(frames[sample['frame']][0]));img=cv2.resize(img,(1280,720)) if img.shape[:2]!=(720,1280) else img;lab=cv2.imread(str(frames[sample['frame']][1]),cv2.IMREAD_UNCHANGED)
        if lab.ndim==3:lab=lab[:,:,0]
        if lab.shape!=(720,1280):lab=cv2.resize(lab,(1280,720),interpolation=cv2.INTER_NEAREST)
        full=np.dstack([img[:,:,::-1],(lab==sample['value'])*255,floor*255,gx,gy]).astype(np.float32);replay=I._resize(full,(320,180)).astype(np.uint8)
        if not np.array_equal(replay,oldF[i]):bad.append({'index':i,'max_diff':int(np.abs(replay.astype(int)-oldF[i].astype(int)).max()),'pixels_changed':int((replay!=oldF[i]).sum())})
        rr,cc=np.where(oldF[i,:,:,3]>63);bbox=np.array([cc.min()/320,rr.min()/180,(cc.max()+1)/320,(rr.max()+1)/180]);G[i,1:5]=bbox;G[i,5:7]=bbox[2:]-bbox[:2]
        x1,y1,x2,y2=bbox*[1280,720,1280,720];w=x2-x1;h=y2-y1;a=max(0,int(x1-w/2));b=max(0,int(y1-h/8));c=min(1280,int(math.ceil(x2+w/2)));d=min(720,int(math.ceil(y2+h/8)))
        crops.append(I._resize(full[b:d,a:c],(96,160)).astype(np.uint8))
        if (i+1)%256==0:print('recovered',i+1,'seconds',round(time.time()-start),'replay_changes',len(bad),flush=True)
    out=root/'data/inout_lab_source_recovery';out.mkdir(exist_ok=True)
    report={'rows':len(ids),'source_replay_mismatches':bad,'ids_sha256':hashlib.sha256(json.dumps(ids.tolist()).encode()).hexdigest(),'labels_sha256':hashlib.sha256(z['y'].tobytes()).hexdigest(),'seconds':time.time()-start,'original_files_modified':False,'source':'Original source RGB/labelmap at1280x720; crop bbox matches existing aligned visible mask; no re-interpolation of low-resolution X'}
    (out/'recovery_report.json').write_text(json.dumps(report,indent=2))
    if bad:raise RuntimeError('Original full-frame replay differs; refusing to export replacement crops')
    np.savez_compressed(out/'recovered_crops.npz',X=np.stack(crops),G=G,ids=ids,y=z['y'],day=z['day']);print(json.dumps(report),flush=True)
_recover_inout_crops_0710()
